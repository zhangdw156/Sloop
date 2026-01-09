import hashlib
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm

from sloop.utils.logger import logger


class GraphSampler:
    """
    GraphSampler (任务架构师)
    职责：从巨大的 API 依赖图谱中，采样出逻辑可行的任务骨架 (TaskSkeleton)。
    原则：
    1. 只关注拓扑结构 (Structure) 和数据流 (Data Flow)。
    2. 不关注用户意图 (Intent) 或具体的剧本内容 (Script)。
    3. 通过覆盖率衰减 (Decay) 机制，确保探索图谱的每一个角落。
    """

    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        # --- 覆盖率记忆模块 ---
        # 记录边被走的次数 key=(u, v, edge_key)
        self.edge_visits = defaultdict(int)
        # 记录节点作为起点的次数
        self.node_starts = defaultdict(int)
        self.total_edges = graph.number_of_edges()

    def reset_coverage(self):
        """重置所有访问记录，重新开始探索"""
        self.edge_visits.clear()
        self.node_starts.clear()
        logger.info("Sampler coverage memory reset.")

    def get_coverage_stats(self) -> Dict:
        """查看当前的图覆盖率"""
        visited_edges = len(self.edge_visits)
        coverage = visited_edges / self.total_edges if self.total_edges > 0 else 0
        return {
            "visited_edges": visited_edges,
            "total_edges": self.total_edges,
            "coverage_ratio": f"{coverage:.2%}",
        }

    # =========================================================================
    # 核心游走逻辑 (Internal Walking Logic)
    # =========================================================================

    def _select_start_node(self) -> str:
        """选择一个起始节点：优先选择出度大于0且使用频率较低的节点"""
        candidates = [n for n in self.graph.nodes() if self.graph.out_degree(n) > 0]
        if not candidates:
            return None

        # 排序策略：启动次数越少越靠前 + 随机扰动 (Exploration)
        # 避免每次都选同一个“最冷门”的节点，增加随机性
        candidates.sort(key=lambda n: self.node_starts[n] + random.random())
        return candidates[0]

    def _get_next_hop(self, current_node: str) -> Tuple:
        """
        根据 Decay 算法选择下一跳。
        返回: (next_node, edge_key, edge_attr)
        """
        successors = list(self.graph.successors(current_node))
        if not successors:
            return None

        candidates = []
        weights = []

        for succ in successors:
            # 获取两点间的所有边 (MultiGraph 可能有多条边)
            edge_data = self.graph.get_edge_data(current_node, succ)
            for key, attr in edge_data.items():
                original_score = attr.get("weight", 0.5)

                # --- 核心衰减公式 ---
                # 1 / (1 + 访问次数)
                # 访问次数越多，权重越低，强迫采样器去探索未走过的路
                visit_count = self.edge_visits[(current_node, succ, key)]
                decay_factor = 1.0 / (1.0 + visit_count)

                final_weight = original_score * decay_factor

                candidates.append((succ, key, attr))
                weights.append(final_weight)

        if not candidates:
            return None

        # 归一化权重并进行加权随机选择
        total_w = sum(weights)
        if total_w == 0:
            probs = [1.0 / len(weights)] * len(weights)
        else:
            probs = [w / total_w for w in weights]

        idx = np.random.choice(len(candidates), p=probs)
        return candidates[idx]

    def _walk_sequential_chain(self, min_len: int, max_len: int) -> Tuple:
        """
        执行一次随机游走，尝试生成一条线性路径。
        这是所有子图生成的基础原子操作。
        返回: (path_nodes, edges_taken, start_node) 或 None
        """
        start_node = self._select_start_node()
        if not start_node:
            return None

        # 随机决定目标长度，保证长短数据分布均匀
        target_len = random.randint(min_len, max_len)

        path_nodes = [start_node]
        edges_taken = []
        visited = {start_node}
        curr = start_node

        for _ in range(target_len - 1):
            hop = self._get_next_hop(curr)
            if not hop:
                break  # 死胡同

            next_node, key, _ = hop
            if next_node in visited:
                break  # 避环

            path_nodes.append(next_node)
            edges_taken.append((curr, next_node, key))
            visited.add(next_node)
            curr = next_node

        # 长度校验
        if len(path_nodes) < min_len:
            # 稍微惩罚导致短路的起点，促使下次换个起点
            self.node_starts[start_node] += 1.0
            return None

        return path_nodes, edges_taken, start_node

    # =========================================================================
    # 公开采样接口 (Public Sampling API)
    # =========================================================================

    def sample_sequential_chain(self, min_len: int = 3, max_len: int = 6) -> Dict:
        """
        [模式 A] 线性链 (Sequential Chain)
        仅包含一条核心逻辑链，没有任何干扰项。
        适合训练：Slot Filling (参数传递), Basic Reasoning.
        """
        result = self._walk_sequential_chain(min_len, max_len)
        if not result:
            return None

        path_nodes, edges_taken, start_node = result
        # 包装成 Skeleton (不在此处更新计数，由 generate_skeletons 统一管理)
        return (
            self._format_skeleton("sequential", path_nodes, edges_taken),
            edges_taken,
            start_node,
        )

    def sample_neighborhood_subgraph(
        self, min_len: int = 2, max_len: int = 4, expansion_ratio: float = 0.5
    ) -> Dict:
        """
        [模式 B] 邻域子图 (Neighborhood Subgraph)
        包含一条 '核心解' (Core Chain) 和若干 '干扰项' (Distractors)。
        适合训练：Tool Selection (工具选择), Robustness (抗干扰).
        """
        # 1. 生成骨架 (Core Chain)
        result = self._walk_sequential_chain(min_len, max_len)
        if not result:
            return None
        path_nodes, edges_taken, start_node = result

        core_tools = set(path_nodes)

        # 2. 扩充子图 (Expansion / Noise Injection)
        # 获取核心链上所有工具的邻居 (作为干扰项候选池)
        candidates = set()
        for tool_name in core_tools:
            candidates.update(self.graph.successors(tool_name))
            candidates.update(self.graph.predecessors(tool_name))

        # 排除骨架本身，剩下的就是干扰项
        distractors_pool = list(candidates - core_tools)

        # 随机选择一部分干扰项加入
        num_extras = int(len(core_tools) * expansion_ratio) + 1
        if len(distractors_pool) > num_extras:
            selected_distractors = random.sample(distractors_pool, num_extras)
        else:
            selected_distractors = distractors_pool

        # 3. 组装 Skeleton
        # all_nodes = Core + Noise
        all_nodes = list(core_tools) + selected_distractors
        random.shuffle(all_nodes)  # 打乱物理顺序，防止模型根据位置作弊

        skeleton = {
            "pattern": "neighborhood_subgraph",
            "nodes": [],
            "edges": [],  # 这里的 edges 只包含 core chain 的有效逻辑
            "meta": {
                "core_chain_nodes": path_nodes,  # 标记答案路径
                "distractor_nodes": selected_distractors,  # 标记干扰项
            },
        }

        # 填充节点
        for node in all_nodes:
            attrs = self.graph.nodes[node]
            skeleton["nodes"].append({
                "name": node,
                "description": attrs.get("desc", ""),
                "category": attrs.get("category", "general"),
                # 在 Skeleton 中显式标记角色，方便 DataFactory 使用
                "role": "core" if node in core_tools else "distractor",
            })

        # 填充边 (只包含 Core Chain 的逻辑依赖)
        for i, (u, v, key) in enumerate(edges_taken):
            edge_data = self.graph.get_edge_data(u, v)[key]
            skeleton["edges"].append({
                "step": i + 1,
                "from": u,
                "to": v,
                "dependency": {
                    "parameter": edge_data.get("parameter"),
                    "relation": "provides_input_for",
                },
            })

        return skeleton, edges_taken, start_node

    # =========================================================================
    # 格式化工具 (Formatting)
    # =========================================================================

    def _format_skeleton(
        self, pattern: str, nodes: List[str], edges: List[Tuple]
    ) -> Dict:
        """将路径格式化为标准的 TaskSkeleton 字典"""
        skeleton = {"pattern": pattern, "nodes": [], "edges": []}
        # 填充节点
        for node in nodes:
            attrs = self.graph.nodes[node]
            skeleton["nodes"].append({
                "name": node,
                "description": attrs.get("desc", ""),
                "category": attrs.get("category", "general"),
                "role": "core",  # 在纯 Chain 模式下，全都是 Core
            })
        # 填充边
        for i, (u, v, key) in enumerate(edges):
            edge_data = self.graph.get_edge_data(u, v)[key]
            skeleton["edges"].append({
                "step": i + 1,
                "from": u,
                "to": v,
                "dependency": {
                    "parameter": edge_data.get("parameter"),
                    "relation": "provides_input_for",
                },
            })
        return skeleton

    # =========================================================================
    # 批量生成入口 (Batch Generator)
    # =========================================================================

    def generate_skeletons(
        self,
        mode: str = "neighborhood",  # Options: 'chain', 'neighborhood'
        count: int = 10,
        min_len: int = 3,
        max_len: int = 6,
        max_retries: int = 500,
        **kwargs,
    ) -> List[Dict]:
        """
        统一的批量生成入口。
        功能：
        1. 调度不同的采样策略 (mode)。
        2. 执行严格去重 (Unique Hash)。
        3. 管理覆盖率更新 (Decay Update)。
        4. 防止死循环 (Max Retries)。
        """
        skeletons = []
        unique_hashes = set()
        fail_streak = 0

        with tqdm(
            total=count, desc=f"Sampling {mode.capitalize()}", unit="skel"
        ) as pbar:
            while len(skeletons) < count:
                # 1. 策略分发
                result = None
                if mode == "chain":
                    result = self.sample_sequential_chain(min_len, max_len)
                elif mode == "neighborhood":
                    ratio = kwargs.get("expansion_ratio", 0.5)
                    result = self.sample_neighborhood_subgraph(min_len, max_len, ratio)

                # 2. 结果处理
                if result:
                    skeleton, edges_taken, start_node = result

                    # 生成唯一指纹
                    # 只要 edges (核心逻辑流) 是独一无二的，这就是一个新任务
                    # 将 edges 排序后 hash，确保无视顺序
                    edge_sig = "|".join(
                        sorted([f"{e['from']}->{e['to']}" for e in skeleton["edges"]])
                    )
                    skel_hash = hashlib.md5(edge_sig.encode()).hexdigest()

                    if skel_hash not in unique_hashes:
                        # === 这是一个新的有效骨架 ===
                        unique_hashes.add(skel_hash)
                        skeletons.append(skeleton)

                        # 只有被收录了，才更新覆盖率计数器
                        self.node_starts[start_node] += 1
                        for e in edges_taken:
                            self.edge_visits[e] += 1

                        fail_streak = 0
                        pbar.update(1)
                    else:
                        fail_streak += 1  # 重复了
                else:
                    fail_streak += 1  # 采样失败(路太短)

                # 3. 防卡死机制
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

    # =========================================================================
    # TODO: [并行采样扩展计划] (Parallel Sampling Extensions)
    # 目前采样器仅实现了 "Sequential" 和 "Neighborhood" 模式。
    # 为了提升 Agent 的并发处理能力，未来建议实现以下两种并行模式：
    #
    # 1. 异构并行 (Heterogeneous Parallel) - "单实体，多维度"
    #    * 场景: 针对同一实体，跨工具获取信息 (如 "查Google股价 + 查Google新闻")。
    #    * 实现: 基于 param_inverted_index 寻找共享参数的工具对。
    #
    # 2. 同构并行 (Homogeneous/Batch Parallel) - "单工具，多实体"
    #    * 场景: 针对列表实体，批量调用同一工具 (如 "查北京、上海、广州的天气")。
    #    * 实现: 随机选择可枚举参数工具，生成 pattern="batch" 的 Skeleton。
    # =========================================================================
