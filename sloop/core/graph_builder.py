import json
import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from sloop.models import ToolDefinition, ToolParameters
from sloop.services import EmbeddingService, LLMService
from sloop.utils.logger import logger


class GraphBuilder:
    def __init__(self):
        # 使用 MultiDiGraph 以支持同一对节点间存在多种关系（如不同参数的依赖）
        self.graph = nx.MultiDiGraph()
        self.tools: Dict[str, ToolDefinition] = {}

        # 1. 初始化 Embedding 服务
        try:
            self.embedding_service = EmbeddingService()
        except Exception as e:
            logger.warning(f"Embedding service failed to init: {e}")
            self.embedding_service = None

        # 2. 初始化 LLM 服务 (修改点：使用封装好的 LLMService)
        self.llm_service = LLMService()
        if not self.llm_service.client:
            logger.warning(
                "LLM service not available. RAG verification will be skipped."
            )

        # 缓存向量
        self.tool_desc_embeddings = {}
        self.param_embeddings = {}

    def load_from_jsonl(self, file_path: str):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Loading tools from {file_path}...")
        processed_lines = 0

        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
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
                            parameters=ToolParameters(
                                **func_data.get("parameters", {})
                            ),
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

        logger.info(
            f"Pre-computing embeddings for {len(self.tools)} tools (Batch size: {batch_size})..."
        )

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
        for name, vec in zip(tool_names, desc_vecs, strict=False):
            if vec is not None:
                self.tool_desc_embeddings[name] = vec
                valid_count += 1

        logger.info(
            f"Successfully embedded {valid_count}/{len(descriptions)} tool descriptions."
        )

        # --- 2. 计算参数描述向量 (Consumer 语义) ---
        all_param_texts = []
        param_indices = []

        for name, tool in self.tools.items():
            if not tool.parameters:
                continue
            for p_name, p_schema in tool.parameters.properties.items():
                text = f"Parameter {p_name}: {p_schema.get('description', '')}"
                all_param_texts.append(text)
                param_indices.append((name, p_name))

        if all_param_texts:
            logger.info(f"Embedding {len(all_param_texts)} parameters...")
            param_vecs = batch_process(all_param_texts, desc="Embedding Params")

            for (t_name, p_name), vec in zip(param_indices, param_vecs, strict=False):
                if vec is not None:
                    if t_name not in self.param_embeddings:
                        self.param_embeddings[t_name] = {}
                    self.param_embeddings[t_name][p_name] = vec

    def _verify_edge_single(self, edge: Dict) -> bool:
        """[新增] 验证单条边的逻辑 (线程安全)"""
        if not self.llm_service.client:
            return False

        p_tool = self.tools[edge["producer"]]
        c_tool = self.tools[edge["consumer"]]
        c_param_desc = c_tool.parameters.properties.get(edge["param"], {}).get(
            "description", ""
        )

        # 针对单条边的 Prompt
        item_text = (
            f"- Producer Tool: {p_tool.name} (Desc: {p_tool.description})\n"
            f"- Consumer Tool: {c_tool.name}\n"
            f"- Consumer Parameter to Fill: '{edge['param']}' (Desc: {c_param_desc})"
        )

        from sloop.prompts.graph import VERIFY_SINGLE_EDGE_SYSTEM_PROMPT

        try:
            resp = self.llm_service.chat_completion(
                messages=[
                    {"role": "system", "content": VERIFY_SINGLE_EDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": item_text},
                ],
                temperature=0.0,  # 验证任务用 0 温度最稳
                response_format={"type": "json_object"},
            )

            if resp:
                import json

                clean_resp = resp.replace("```json", "").replace("```", "").strip()
                result = json.loads(clean_resp)
                # 假设返回 {"valid": true/false}
                return result.get("valid", False)
        except Exception as e:
            logger.warning(
                f"Edge verification failed for {p_tool.name}->{c_tool.name}: {e}"
            )

        return False

    def _verify_edges_concurrent(
        self, candidates: List[Dict], max_workers=20
    ) -> List[Dict]:
        """使用多线程并发验证边"""
        if not candidates:
            return []

        logger.info(
            f"Verifying {len(candidates)} edges concurrently (Max Workers: {max_workers})..."
        )
        valid_edges = []

        import concurrent.futures

        # 使用 ThreadPoolExecutor 并发调用
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            # future -> edge_info 映射
            future_to_edge = {
                executor.submit(self._verify_edge_single, edge): edge
                for edge in candidates
            }

            # 使用 tqdm 显示进度
            for future in tqdm(
                concurrent.futures.as_completed(future_to_edge),
                total=len(candidates),
                desc="LLM Verify",
            ):
                edge = future_to_edge[future]
                try:
                    is_valid = future.result()
                    if is_valid:
                        valid_edges.append(edge)
                except Exception as exc:
                    logger.error(f"Verification thread generated an exception: {exc}")

        return valid_edges

    def build(
        self,
        recall_threshold: float = 0.68,  # 建议调高到 0.68 或 0.70
        auto_accept_threshold: float = 0.88,  # 建议调高高置信度门槛
        top_k: int = 5,  # 新增：每个参数只取前 5 个最像的候选
        enable_llm_verify: bool = True,
        prune_isolates: bool = True,  # 新增参数
    ):
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

        self._auto_categorize_concurrent(max_workers=50)
        self._precompute_embeddings()

        logger.info(
            f"Building graph (Recall Threshold: {recall_threshold}, Auto-Accept: {auto_accept_threshold})..."
        )

        # 2. 添加节点
        for name, tool in self.tools.items():
            self.graph.add_node(
                name,
                desc=tool.description,
                category=tool.category,
                parameters=tool.parameters.dict(),
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
        logger.info(
            f"Filtering candidates (Top-{top_k} per param & > {recall_threshold})..."
        )

        rows = []
        cols = []

        # 不直接用 np.where，而是对每一列 (Consumer Param) 取 Top-K
        # C.shape[0] 是参数的总数
        num_consumers = similarity_matrix.shape[1]

        # 使用 tqdm 显示筛选进度
        for c_idx in tqdm(range(num_consumers), desc="Filtering Top-K", unit="param"):
            # 获取这一列的所有分数
            scores = similarity_matrix[:, c_idx]

            # 1. 只有大于阈值的才考虑 (初步过滤)
            # 使用 argwhere 找到大于阈值的索引
            valid_indices = np.where(scores > recall_threshold)[0]

            if len(valid_indices) == 0:
                continue

            # 2. 如果合格的数量超过 Top-K，只取分最高的 Top-K
            if len(valid_indices) > top_k:
                # 获取这些合格者的分数
                valid_scores = scores[valid_indices]
                # 找到分数最高的 top_k 个的局部索引
                # argpartition 比 argsort 快，用于非严格排序的 Top-K
                top_k_local_indices = np.argpartition(valid_scores, -top_k)[-top_k:]
                # 映射回全局索引
                best_producer_indices = valid_indices[top_k_local_indices]
            else:
                best_producer_indices = valid_indices

            # 记录结果
            for r_idx in best_producer_indices:
                rows.append(r_idx)
                cols.append(c_idx)

        rows = np.array(rows)
        cols = np.array(cols)

        logger.info(
            f"Reduced candidates to {len(rows)} edges using Top-{top_k} strategy."
        )

        # 5. 分层处理与验证
        candidates_to_verify = []  # 待 LLM 验证的列表
        edges_to_add = []  # 最终要添加的边

        stats = {"direct": 0, "verified": 0, "rejected": 0}

        # 遍历所有候选
        for r, c in zip(rows, cols, strict=False):
            producer_name = producer_names[r]
            consumer_name, param_name = consumer_map[c]

            if producer_name == consumer_name:
                continue

            score = float(similarity_matrix[r, c])
            edge_info = {
                "producer": producer_name,
                "consumer": consumer_name,
                "param": param_name,
                "score": score,
            }

            # 策略分支
            if score >= auto_accept_threshold:
                # 高置信度：直接通过
                edges_to_add.append(edge_info)
                stats["direct"] += 1
            elif enable_llm_verify and self.llm_service:
                # 中置信度：加入 LLM 验证队列
                candidates_to_verify.append(edge_info)
            # 如果没开 LLM 且分数不够高，或者分数低于 recall 但高于 np.where (理论上不可能)，则通过(如果没开LLM)或丢弃
            # 这里逻辑：如果 enable_llm_verify=False，则 recall_threshold 即为最终门槛
            elif not enable_llm_verify:
                edges_to_add.append(edge_info)
                stats["direct"] += 1

        # 6. 执行批量 LLM 验证
        if candidates_to_verify:
            valid_batch = self._verify_edges_concurrent(
                candidates_to_verify, max_workers=50
            )
            edges_to_add.extend(valid_batch)
            stats["verified"] += len(valid_batch)
            stats["rejected"] += len(candidates_to_verify) - len(valid_batch)

        # 7. 最终添加边到图谱
        logger.info(f"Adding {len(edges_to_add)} edges to graph...")
        for edge in edges_to_add:
            self.graph.add_edge(
                edge["producer"],
                edge["consumer"],
                key=f"{edge['producer']}->{edge['consumer']}:{edge['param']}",
                relation="provides_parameter",
                parameter=edge["param"],
                weight=edge["score"],
            )

        logger.info("Graph build complete.")
        logger.info(
            f"Stats: High Conf (Direct): {stats['direct']}, LLM Verified: {stats['verified']}, LLM Rejected: {stats['rejected']}"
        )

        if prune_isolates:
            self.prune_graph(min_component_size=2)

        logger.info(
            f"Final Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges."
        )

        return self.graph

    def visualize(self, output_path: str = "data/tool_graph_vis.png"):
        if self.graph.number_of_nodes() == 0:
            return
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(self.graph, k=0.6, iterations=50)
        nx.draw_networkx_nodes(
            self.graph, pos, node_size=800, node_color="#a8d5e2", alpha=0.9
        )
        nx.draw_networkx_edges(
            self.graph, pos, edge_color="gray", alpha=0.4, arrowsize=10
        )
        nx.draw_networkx_labels(self.graph, pos, font_size=7)
        plt.title("Semantic Tool Graph")
        plt.axis("off")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Graph saved to {output_path}")

    def export_graph_json(self, output_path: str = "data/graph.json"):
        data = nx.node_link_data(self.graph)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Graph exported to {output_path}")

    def export_graphml(self, output_path: str = "data/graph.graphml"):
        G_export = self.graph.copy()
        for _node, data in G_export.nodes(data=True):
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    data[k] = str(v)
        try:
            nx.write_graphml(G_export, output_path)
            logger.info(f"Graph exported to {output_path} (Ready for Cytoscape)")
        except Exception as e:
            logger.error(f"Failed to export GraphML: {e}")

    def save_checkpoint(self, path: str = "data/graph_checkpoint.pkl"):
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_data = {
            "graph": self.graph,
            "tools": self.tools,
            "desc_embeddings": self.tool_desc_embeddings,
            "param_embeddings": self.param_embeddings,
        }
        try:
            with open(save_path, "wb") as f:
                pickle.dump(checkpoint_data, f)
            logger.info(f"Graph checkpoint saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, path: str = "data/graph_checkpoint.pkl") -> bool:
        load_path = Path(path)
        if not load_path.exists():
            logger.warning(f"Checkpoint not found at {load_path}")
            return False
        try:
            with open(load_path, "rb") as f:
                data = pickle.load(f)
            self.graph = data.get("graph", nx.MultiDiGraph())
            self.tools = data.get("tools", {})
            self.tool_desc_embeddings = data.get("desc_embeddings", {})
            self.param_embeddings = data.get("param_embeddings", {})
            logger.info(f"Successfully loaded graph from {load_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def prune_graph(self, min_component_size: int = 2):
        """
        剪枝：移除过小的连通分量。
        默认 min_component_size=2，意味着移除所有孤立节点（单点）。
        """
        if self.graph.number_of_nodes() == 0:
            return

        original_count = self.graph.number_of_nodes()

        # 获取所有弱连通分量 (Weakly Connected Components)
        # 弱连通意味着把有向边看作无向边时是连通的
        components = list(nx.weakly_connected_components(self.graph))

        nodes_to_remove = []
        for comp in components:
            if len(comp) < min_component_size:
                nodes_to_remove.extend(comp)

        if nodes_to_remove:
            self.graph.remove_nodes_from(nodes_to_remove)
            logger.info(
                f"Pruned graph: Removed {len(nodes_to_remove)} nodes belonging to components size < {min_component_size}."
            )
            logger.info(
                f"Node count: {original_count} -> {self.graph.number_of_nodes()}"
            )
        else:
            logger.info("No nodes pruned.")

    def show(self):
        """
        在控制台展示图谱的详细统计信息 (Rich Table)
        """
        if self.graph.number_of_nodes() == 0:
            logger.warning("Graph is empty, nothing to show.")
            return

        console = Console()

        # --- 1. Basic Stats ---
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        density = nx.density(self.graph)

        # --- 2. Component Analysis ---
        components = list(nx.weakly_connected_components(self.graph))
        num_components = len(components)
        largest_comp_size = max(len(c) for c in components) if components else 0
        num_isolates = sum(1 for c in components if len(c) == 1)

        # --- 3. Degree Distribution (Top 5) ---
        in_degrees = sorted(self.graph.in_degree, key=lambda x: x[1], reverse=True)[:5]
        out_degrees = sorted(self.graph.out_degree, key=lambda x: x[1], reverse=True)[
            :5
        ]

        # --- 4. Top Parameters ---
        param_counts = {}
        for _, _, data in self.graph.edges(data=True):
            p = data.get("parameter", "unknown")
            param_counts[p] = param_counts.get(p, 0) + 1
        top_params = sorted(param_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # --- Build Table ---
        table = Table(
            title="Semantic Tool Graph Stats",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Nodes", str(num_nodes))
        table.add_row("Edges", str(num_edges))
        table.add_row("Density", f"{density:.5f}")
        table.add_section()
        table.add_row("Components", str(num_components))
        table.add_row("Largest Component", str(largest_comp_size))
        table.add_row("Isolates (Singletons)", str(num_isolates))
        table.add_section()

        # Format lists for display
        top_producers = ", ".join([f"{n}({d})" for n, d in out_degrees])
        top_consumers = ", ".join([f"{n}({d})" for n, d in in_degrees])
        top_params_str = ", ".join([f"{k}({v})" for k, v in top_params])

        table.add_row("Top Producers (Out-Degree)", top_producers)
        table.add_row("Top Consumers (In-Degree)", top_consumers)
        table.add_row("Top Parameters", top_params_str)

        console.print(table)

    def _categorize_single_tool(self, tool: ToolDefinition, category_pool: set) -> str:
        """[新增] 对单个工具进行分类 (线程安全)"""
        if not self.llm_service.client:
            return None

        # 每次取当前的 pool 快照
        existing_cats_str = ", ".join(sorted(category_pool))

        tool_info = f"- Name: {tool.name}\n  Desc: {tool.description[:200]}"

        from sloop.prompts.graph import AUTO_CATEGORIZE_SINGLE_SYSTEM_PROMPT

        system_prompt = AUTO_CATEGORIZE_SINGLE_SYSTEM_PROMPT.format(
            existing_cats_str=existing_cats_str
        )

        try:
            resp = self.llm_service.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": tool_info},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            if resp:
                import json

                clean_resp = resp.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_resp)
                cat = data.get("category")
                if cat:
                    return cat.strip().title()
        except Exception as e:
            logger.warning(f"Categorization failed for {tool.name}: {e}")

        return None

    def _auto_categorize_concurrent(self, max_workers=20):
        """[修改] 多线程动态分类"""
        if not self.llm_service or not self.llm_service.client:
            return

        tools_to_process = [t for t in self.tools.values() if t.category == "general"]
        if not tools_to_process:
            return

        # 初始池
        category_pool = {
            "Sports",
            "Finance",
            "Weather",
            "Utilities",
            "Entertainment",
            "Shopping",
            "Education",
        }

        logger.info(f"Categorizing {len(tools_to_process)} tools concurrently...")

        import concurrent.futures

        # 注意：这里有个并发写 pool 的问题
        # 虽然 set 不是线程安全的，但在 Python GIL 下简单的 add 操作通常没问题。
        # 且即使有一两个 update 冲突了，对于分类池的影响微乎其微（顶多 LLM 没看到最新的那个 tag）。
        # 为了稳妥，我们可以把 pool 的更新放在主线程。

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 这里的 pool 是传值（快照），随着循环进行，后面的任务可能会拿到旧的 pool
            # 但这对 "Reuse First" 影响不大，因为初始池已经够大了。
            future_to_tool = {
                executor.submit(self._categorize_single_tool, tool, category_pool): tool
                for tool in tools_to_process
            }

            for future in tqdm(
                concurrent.futures.as_completed(future_to_tool),
                total=len(tools_to_process),
                desc="Auto Categorizing",
            ):
                tool = future_to_tool[future]
                try:
                    cat = future.result()
                    if cat:
                        tool.category = cat
                        category_pool.add(cat)  # 更新池子
                except Exception as exc:
                    logger.error(f"Categorization thread exception: {exc}")

        logger.info(
            f"Categorization complete. Final Pool ({len(category_pool)}): {list(category_pool)}"
        )
