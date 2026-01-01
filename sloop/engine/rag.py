"""
向量检索引擎

实现基于 FAISS 的工具向量检索，用于 RAG 增强的采样逻辑。
"""

import json
import os
from pathlib import Path
from typing import List

import faiss
import numpy as np
from tqdm import tqdm

from sloop.config import get_settings
from sloop.models import ToolDefinition
from sloop.utils.logger import logger


class ToolRetrievalEngine:
    """
    工具向量检索引擎

    使用 FAISS 构建工具向量索引，支持语义搜索相似工具。
    """

    def __init__(self, cache_dir: str = ".cache"):
        """
        初始化检索引擎

        参数:
            cache_dir: 缓存目录路径，相对于当前工作目录
        """
        self.cache_dir = Path(cache_dir)
        self.index_path = self.cache_dir / "tool_index.faiss"
        self.names_path = self.cache_dir / "tool_names.json"

        # 确保缓存目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 初始化属性
        self.index = None
        self.tool_names = []

        # 尝试加载现有索引
        self._load_index()

        # 获取配置
        self.settings = get_settings()

    def _load_index(self):
        """加载现有索引"""
        if self.index_path.exists() and self.names_path.exists():
            try:
                # 加载 FAISS 索引
                self.index = faiss.read_index(str(self.index_path))

                # 加载工具名称映射
                with open(self.names_path, 'r', encoding='utf-8') as f:
                    self.tool_names = json.load(f)

                logger.info(f"Loaded index successfully: {len(self.tool_names)} tools")

            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
                self.index = None
                self.tool_names = []
        else:
            logger.info("No existing index files found")

    def _save_index(self):
        """保存索引到磁盘"""
        if self.index is not None and self.tool_names:
            try:
                # 保存 FAISS 索引
                faiss.write_index(self.index, str(self.index_path))

                # 保存工具名称映射
                with open(self.names_path, 'w', encoding='utf-8') as f:
                    json.dump(self.tool_names, f, ensure_ascii=False, indent=2)

                logger.info(f"Index saved successfully: {self.index_path}, {self.names_path}")

            except Exception as e:
                logger.error(f"Failed to save index: {e}")

    def _get_embedding(self, text: str | List[str]) -> List[float] | List[List[float]]:
        """
        获取文本的向量表示

        参数:
            text: 输入文本或文本列表

        返回:
            向量列表或向量列表的列表
        """
        import litellm

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = litellm.embedding(
                    model=f"{self.settings.embedding_provider}/{self.settings.embedding_model}",
                    input=text,
                    api_key=self.settings.embedding_api_key,
                    api_base=self.settings.embedding_base_url,
                    encoding_format="float",
                )

                if response and response.data:
                    if isinstance(text, str):
                        # 单条输入
                        if len(response.data) > 0:
                            item = response.data[0]
                            if hasattr(item, 'embedding'):
                                return item.embedding
                            elif isinstance(item, dict) and 'embedding' in item:
                                return item['embedding']
                            else:
                                # 假设 item 就是向量列表
                                return item
                    else:
                        # 批量输入
                        embeddings = []
                        for item in response.data:
                            if hasattr(item, 'embedding'):
                                embeddings.append(item.embedding)
                            elif isinstance(item, dict) and 'embedding' in item:
                                embeddings.append(item['embedding'])
                            else:
                                # 假设 item 就是向量列表
                                embeddings.append(item)
                        return embeddings

            except Exception as e:
                logger.warning(f"Embedding call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    continue

        # 失败时抛出异常
        raise RuntimeError("Embedding call failed after all retries")

    def build(self, tools: List[ToolDefinition], force: bool = False):
        """
        构建工具向量索引

        参数:
            tools: 工具定义列表
            force: 是否强制重新构建
        """
        # 检查是否需要跳过构建
        if not force and self.index is not None and self.tool_names and len(self.tool_names) == len(tools):
            logger.info("Index already exists and tool count matches, skipping build (use force=True to rebuild)")
            return
        elif not force and self.index is not None and self.tool_names and len(self.tool_names) != len(tools):
            logger.warning(f"Tool count mismatch: index has {len(self.tool_names)} tools, input has {len(tools)} tools, rebuilding...")

        logger.info(f"Starting to build index: {len(tools)} tools")

        # 准备数据
        texts = []
        self.tool_names = []

        for tool in tools:
            # 构造语义文本
            params_str = json.dumps(tool.parameters.model_dump(), ensure_ascii=False)
            text = f"name: {tool.name} description: {tool.description} params: {params_str}"
            texts.append(text)
            self.tool_names.append(tool.name)

        logger.info("Generating embeddings...")

        # 批量生成向量（分批处理，避免 API 限制）
        batch_size = 10
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]

            # 批量调用 embedding API
            batch_embeddings = self._get_embedding(batch_texts)
            all_embeddings.extend(batch_embeddings)

        # 转换为 numpy 数组
        embeddings = np.array(all_embeddings, dtype=np.float32)

        logger.info(f"Embeddings shape: {embeddings.shape}")

        # 构建 FAISS 索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        logger.info(f"Index build completed: {len(self.tool_names)} tools")

        # 保存索引
        self._save_index()

    def search(self, query_tool: ToolDefinition, top_k: int = 10) -> List[str]:
        """
        搜索相似的工具

        参数:
            query_tool: 查询工具定义
            top_k: 返回的相似工具数量

        返回:
            相似工具名称列表
        """
        if self.index is None or not self.tool_names:
            logger.error("Index not built, cannot search")
            return []

        # 构造查询文本
        params_str = json.dumps(query_tool.parameters.model_dump(), ensure_ascii=False)
        query_text = f"name: {query_tool.name} description: {query_tool.description} params: {params_str}"

        # 获取查询向量
        query_embedding = self._get_embedding(query_text)
        query_vector = np.array([query_embedding], dtype=np.float32)

        # 搜索相似向量
        distances, indices = self.index.search(query_vector, min(top_k, len(self.tool_names)))

        # 返回工具名称
        results = []
        for idx in indices[0]:
            if idx < len(self.tool_names):
                results.append(self.tool_names[idx])

        return results
