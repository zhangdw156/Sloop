import hashlib
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm

from sloop.models import (
    Dependency,
    SkeletonEdge,
    SkeletonMeta,
    SkeletonNode,
    TaskSkeleton,
)
from sloop.utils.logger import logger


class GraphSampler:
    """
    GraphSampler (任务架构师)
    职责：从巨大的 API 依赖图谱中，采样出逻辑可行的任务骨架 (TaskSkeleton)。
    """

    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        # --- 覆盖率记忆模块 ---
        self.edge_visits = defaultdict(int)
        self.node_starts = defaultdict(int)
        self.total_edges = graph.number_of_edges()

    def reset_coverage(self):
        self.edge_visits.clear()
        self.node_starts.clear()
        logger.info("Sampler coverage memory reset.")

    def get_coverage_stats(self) -> Dict:
        visited_edges = len(self.edge_visits)
        coverage = visited_edges / self.total_edges if self.total_edges > 0 else 0
        return {
            "visited_edges": visited_edges,
            "total_edges": self.total_edges,
            "coverage_ratio": f"{coverage:.2%}",
        }

    # =========================================================================
    # 核心游走逻辑 (Internal Walking Logic) - 保持不变
    # =========================================================================

    def _select_start_node(self) -> str:
        candidates = [n for n in self.graph.nodes() if self.graph.out_degree(n) > 0]
        if not candidates:
            return None
        candidates.sort(key=lambda n: self.node_starts[n] + random.random())
        return candidates[0]

    def _get_next_hop(self, current_node: str) -> Tuple:
        successors = list(self.graph.successors(current_node))
        if not successors:
            return None

        candidates = []
        weights = []

        for succ in successors:
            edge_data = self.graph.get_edge_data(current_node, succ)
            for key, attr in edge_data.items():
                original_score = attr.get("weight", 0.5)
                visit_count = self.edge_visits[(current_node, succ, key)]
                decay_factor = 1.0 / (1.0 + visit_count)
                final_weight = original_score * decay_factor

                candidates.append((succ, key, attr))
                weights.append(final_weight)

        if not candidates:
            return None

        total_w = sum(weights)
        if total_w == 0:
            probs = [1.0 / len(weights)] * len(weights)
        else:
            probs = [w / total_w for w in weights]

        idx = np.random.choice(len(candidates), p=probs)
        return candidates[idx]

    def _walk_sequential_chain(self, min_len: int, max_len: int) -> Tuple:
        """执行游走，返回原始数据，不负责格式化"""
        start_node = self._select_start_node()
        if not start_node:
            return None

        target_len = random.randint(min_len, max_len)
        path_nodes = [start_node]
        edges_taken = []
        visited = {start_node}
        curr = start_node

        for _ in range(target_len - 1):
            hop = self._get_next_hop(curr)
            if not hop:
                break

            next_node, key, _ = hop
            if next_node in visited:
                break

            path_nodes.append(next_node)
            edges_taken.append((curr, next_node, key))
            visited.add(next_node)
            curr = next_node

        if len(path_nodes) < min_len:
            self.node_starts[start_node] += 1.0
            return None

        return path_nodes, edges_taken, start_node

    # =========================================================================
    # 格式化工具 (Formatting) - 修改为返回 Pydantic 对象
    # =========================================================================

    def _format_skeleton(
        self, pattern: str, nodes: List[str], edges: List[Tuple], meta_info: Dict = None
    ) -> TaskSkeleton:
        """将路径格式化为 TaskSkeleton 对象"""

        # 1. 构建 Nodes
        skel_nodes = []
        distractors = set(meta_info.get("distractor_nodes", [])) if meta_info else set()

        for node in nodes:
            attrs = self.graph.nodes[node]
            role = "distractor" if node in distractors else "core"

            skel_nodes.append(
                SkeletonNode(
                    name=node,
                    description=attrs.get("desc", ""),
                    category=attrs.get("category", "general"),
                    role=role,
                )
            )

        # 2. 构建 Edges
        skel_edges = []
        for i, (u, v, key) in enumerate(edges):
            edge_data = self.graph.get_edge_data(u, v)[key]
            skel_edges.append(
                SkeletonEdge(
                    step=i + 1,
                    from_tool=u,  # Pydantic 会自动映射到 alias "from"
                    to_tool=v,  # Pydantic 会自动映射到 alias "to"
                    dependency=Dependency(
                        parameter=edge_data.get("parameter"),
                        relation="provides_input_for",
                    ),
                )
            )

        # 3. 构建 Meta
        meta_obj = None
        if meta_info:
            meta_obj = SkeletonMeta(
                core_chain_nodes=meta_info.get("core_chain_nodes", []),
                distractor_nodes=meta_info.get("distractor_nodes", []),
            )

        # 4. 返回完整对象
        return TaskSkeleton(
            pattern=pattern, nodes=skel_nodes, edges=skel_edges, meta=meta_obj
        )

    # =========================================================================
    # 公开采样接口 (Public Sampling API)
    # =========================================================================

    def sample_sequential_chain(
        self, min_len: int = 3, max_len: int = 6
    ) -> Tuple[TaskSkeleton, List, str]:
        """[模式 A] 线性链"""
        result = self._walk_sequential_chain(min_len, max_len)
        if not result:
            return None

        path_nodes, edges_taken, start_node = result

        # 构造对象
        skeleton = self._format_skeleton("sequential", path_nodes, edges_taken)

        return skeleton, edges_taken, start_node

    def sample_neighborhood_subgraph(
        self, min_len: int = 2, max_len: int = 4, expansion_ratio: float = 0.5
    ) -> Tuple[TaskSkeleton, List, str]:
        # 1. 生成骨架
        result = self._walk_sequential_chain(min_len, max_len)
        if not result:
            return None
        path_nodes, edges_taken, start_node = result
        core_tools = set(path_nodes)

        # 2. 计算目标噪音数量
        num_extras = int(len(core_tools) * expansion_ratio) + 1
        selected_distractors = []

        # --- 策略 A: 优先尝试邻居 (Hard Negatives) ---
        candidates = set()
        for tool_name in core_tools:
            candidates.update(self.graph.successors(tool_name))
            candidates.update(self.graph.predecessors(tool_name))

        # 排除核心链本身
        hard_pool = list(candidates - core_tools)

        if hard_pool:
            # 如果邻居比需要的少，就全拿走；否则随机选
            take_k = min(len(hard_pool), num_extras)
            selected_distractors.extend(random.sample(hard_pool, take_k))

        # --- 策略 B: 补足随机噪音 (Random/Easy Negatives) ---
        # 如果策略 A 没凑够数量，从全图中随机抽
        needed = num_extras - len(selected_distractors)
        if needed > 0:
            all_nodes = list(self.graph.nodes())
            # 排除掉核心链和已经选中的干扰项
            exclude_set = core_tools.union(set(selected_distractors))
            random_pool = [n for n in all_nodes if n not in exclude_set]

            if len(random_pool) >= needed:
                selected_distractors.extend(random.sample(random_pool, needed))
            else:
                # 极端情况：图太小了，把剩下的全加上
                selected_distractors.extend(random_pool)

        # 3. 组装节点列表
        all_nodes = list(core_tools) + selected_distractors
        random.shuffle(all_nodes)

        # 4. 构造对象
        meta_info = {
            "core_chain_nodes": path_nodes,
            "distractor_nodes": selected_distractors,
        }

        # 注意：edges_taken 只包含核心链的边，干扰项是孤立的点（在当前子图中无连接）或者仅作为上下文存在
        skeleton = self._format_skeleton(
            pattern="neighborhood_subgraph",
            nodes=all_nodes,
            edges=edges_taken,
            meta_info=meta_info,
        )

        return skeleton, edges_taken, start_node

    # =========================================================================
    # 批量生成入口
    # =========================================================================

    def generate_skeletons(
        self,
        mode: str = "neighborhood",
        count: int = 10,
        min_len: int = 3,
        max_len: int = 6,
        max_retries: int = 500,
        **kwargs,
    ) -> List[TaskSkeleton]:
        """
        统一批量生成入口，返回 TaskSkeleton 对象列表。
        """
        skeletons = []
        unique_hashes = set()
        fail_streak = 0

        with tqdm(
            total=count, desc=f"Sampling {mode.capitalize()}", unit="skel"
        ) as pbar:
            while len(skeletons) < count:
                result = None
                if mode == "chain":
                    result = self.sample_sequential_chain(min_len, max_len)
                elif mode == "neighborhood":
                    ratio = kwargs.get("expansion_ratio", 0.5)
                    result = self.sample_neighborhood_subgraph(min_len, max_len, ratio)

                if result:
                    skeleton, edges_taken, start_node = result

                    # 生成唯一指纹 (使用对象方法)
                    edge_sig = skeleton.get_edges_signature()
                    skel_hash = hashlib.md5(edge_sig.encode()).hexdigest()

                    if skel_hash not in unique_hashes:
                        unique_hashes.add(skel_hash)
                        skeletons.append(skeleton)

                        # 更新覆盖率
                        self.node_starts[start_node] += 1
                        for e in edges_taken:
                            self.edge_visits[e] += 1

                        fail_streak = 0
                        pbar.update(1)
                    else:
                        fail_streak += 1
                else:
                    fail_streak += 1

                if fail_streak >= max_retries:
                    logger.warning(
                        f"Hit max retries ({max_retries}). Graph might be saturated."
                    )
                    break

        stats = self.get_coverage_stats()
        logger.info(
            f"Generated {len(skeletons)} skeletons. Coverage: {stats['coverage_ratio']}"
        )
        return skeletons
