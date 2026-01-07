import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm
from sloop.services import RemoteEmbeddingService, LLMService
from sklearn.metrics.pairwise import cosine_similarity

from sloop.models import ToolDefinition, ToolParameters
from sloop.utils.logger import logger

class ToolGraphBuilder:
    def __init__(self):
        # 使用 MultiDiGraph 以支持同一对节点间存在多种关系（如不同参数的依赖）
        self.graph = nx.MultiDiGraph()
        self.tools: Dict[str, ToolDefinition] = {}

        # 1. 初始化 Embedding 服务
        try:
            self.embedding_service = RemoteEmbeddingService()
        except Exception as e:
            logger.warning(f"Embedding service failed to init: {e}")
            self.embedding_service = None

        # 2. 初始化 LLM 服务 (修改点：使用封装好的 LLMService)
        self.llm_service = LLMService()
        if not self.llm_service.client:
            logger.warning("LLM service not available. RAG verification will be skipped.")
        
        # 缓存向量
        self.tool_desc_embeddings = {}
        self.param_embeddings = {}

    def load_from_jsonl(self, file_path: str):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Loading tools from {file_path}...")
        processed_lines = 0

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    tools_list = []

                    # 兼容 tools 字段为字符串的情况
                    if "tools" in data and isinstance(data["tools"], str):
                        try:
                            tools_list = json.loads(data["tools"])
                        except json.JSONDecodeError:
                            continue
                    else:
                        tools_list = [data]

                    for tool_dict in tools_list:
                        func_data = tool_dict.get("function", tool_dict)
                        tool = ToolDefinition(
                            name=func_data.get("name"),
                            description=func_data.get("description", ""),
                            parameters=ToolParameters(**func_data.get("parameters", {}))
                        )
                        if tool.name and tool.name not in self.tools:
                            self.tools[tool.name] = tool
                    processed_lines += 1
                except Exception as e:
                    logger.error(f"Error processing line: {e}")

        logger.info(f"Loaded {len(self.tools)} tools from {processed_lines} records.")

    def _precompute_embeddings(self, batch_size=64):
        """批量计算向量，构建语义索引（增加分批处理）"""
        if not self.embedding_service:
            return

        logger.info(f"Pre-computing embeddings for {len(self.tools)} tools (Batch size: {batch_size})...")

        # --- 1. 计算工具描述向量 (Producer 语义) ---
        tool_names = list(self.tools.keys())
        descriptions = [f"{t.name}: {t.description}" for t in self.tools.values()]

        # 分批处理函数
        def batch_process(items, desc="Embedding"):
            results = []
            with tqdm(total=len(items), desc=desc, unit="item") as pbar:
                for i in range(0, len(items), batch_size):
                    batch = items[i : i + batch_size]
                    try:
                        # 调用 API
                        vectors = self.embedding_service.get_embedding(batch)
                        results.extend(vectors)
                    except Exception as e:
                        logger.error(f"Batch embedding failed at index {i}: {e}")
                        results.extend([None] * len(batch))
                    pbar.update(len(batch))
            return results

        # 执行分批计算
        desc_vecs = batch_process(descriptions, desc="Embedding Descriptions")

        # 存入缓存
        valid_count = 0
        for name, vec in zip(tool_names, desc_vecs):
            if vec is not None:
                self.tool_desc_embeddings[name] = vec
                valid_count += 1

        logger.info(f"Successfully embedded {valid_count}/{len(descriptions)} tool descriptions.")

        # --- 2. 计算参数描述向量 (Consumer 语义) ---
        all_param_texts = []
        param_indices = []

        for name, tool in self.tools.items():
            if not tool.parameters: continue
            for p_name, p_schema in tool.parameters.properties.items():
                text = f"Parameter {p_name}: {p_schema.get('description', '')}"
                all_param_texts.append(text)
                param_indices.append((name, p_name))

        if all_param_texts:
            logger.info(f"Embedding {len(all_param_texts)} parameters...")
            param_vecs = batch_process(all_param_texts, desc="Embedding Params")

            for (t_name, p_name), vec in zip(param_indices, param_vecs):
                if vec is not None:
                    if t_name not in self.param_embeddings:
                        self.param_embeddings[t_name] = {}
                    self.param_embeddings[t_name][p_name] = vec

    def _verify_batch_edges(self, candidates: List[Dict]) -> List[Dict]:
        """使用 LLM 批量验证边"""
        if not self.llm_service.client or not candidates:
            return []

        # 构造 Prompt
        prompt_items = []
        for idx, item in enumerate(candidates):
            p_tool = self.tools[item['producer']]
            c_tool = self.tools[item['consumer']]
            c_param_desc = c_tool.parameters.properties.get(item['param'], {}).get('description', '')
            
            prompt_items.append(
                f"ID {idx}:\n"
                f" - Producer Tool: {p_tool.name} (Desc: {p_tool.description})\n"
                f" - Consumer Tool: {c_tool.name}\n"
                f" - Consumer Parameter to Fill: '{item['param']}' (Desc: {c_param_desc})\n"
            )

        items_text = "\n".join(prompt_items)
        
        system_prompt = """You are an expert in API integration. 
Your task is to verify if the output of the 'Producer Tool' can logically and semantically serve as the input for the 'Consumer Parameter'.
Ignore weak or coincidental connections (e.g., both involving 'date' but for unrelated events).
Focus on business logic flow (e.g., search product -> get details -> get reviews).

Return a JSON object with a single key 'valid_ids' containing the list of integer IDs for valid connections.
Example: {"valid_ids": [1, 3, 5]}"""

        user_prompt = f"Verify the following connections:\n\n{items_text}"

        content = self.llm_service.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        if content:
            try:
                result = json.loads(content)
                valid_ids = set(result.get("valid_ids", []))
                return [candidates[i] for i in valid_ids if i < len(candidates)]
            except json.JSONDecodeError:
                logger.error("Failed to parse LLM JSON response")
        
        return []

    def build(self, 
              recall_threshold: float = 0.60, 
              auto_accept_threshold: float = 0.85,
              enable_llm_verify: bool = True):
        """
        构建增强型知识图谱 (RAG 增强版)
        
        策略：
        1. 相似度 > auto_accept_threshold: 直接通过 (高置信度)
        2. recall_threshold < 相似度 < auto_accept_threshold: LLM 验证 (模糊区间)
        3. 相似度 < recall_threshold: 丢弃
        """
        if not self.tools:
            logger.warning("No tools loaded.")
            return self.graph

        self._precompute_embeddings()

        logger.info(f"Building graph (Recall Threshold: {recall_threshold}, Auto-Accept: {auto_accept_threshold})...")

        # 2. 添加节点
        for name, tool in self.tools.items():
            self.graph.add_node(
                name,
                desc=tool.description,
                category=tool.category,
                parameters=tool.parameters.dict()
            )

        # 3. 准备矩阵数据
        logger.info("Preparing matrices...")
        producer_names = []
        producer_matrix = []
        for name, vec in self.tool_desc_embeddings.items():
            producer_names.append(name)
            producer_matrix.append(vec)

        consumer_map = [] 
        consumer_matrix = []
        for tool_name, params in self.param_embeddings.items():
            for param_name, vec in params.items():
                consumer_map.append((tool_name, param_name))
                consumer_matrix.append(vec)

        if not producer_matrix or not consumer_matrix:
            return self.graph

        P = np.array(producer_matrix)
        C = np.array(consumer_matrix)

        logger.info("Computing similarity matrix...")
        similarity_matrix = cosine_similarity(P, C)

        # 4. 筛选候选集 (使用较低的 recall_threshold)
        rows, cols = np.where(similarity_matrix
