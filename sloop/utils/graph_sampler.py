import hashlib
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm  # 引入 tqdm 显示进度

from sloop.utils.logger import logger


class GraphSampler:
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        # 覆盖率记忆
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

    def _select_start_node(self) -> str:
        candidates = [n for n in self.graph.nodes() if self.graph.out_degree(n) > 0]
        if not candidates:
            return None
        # 排序策略：启动次数越少越靠前 + 随机扰动
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
                # 衰减公式
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

    def sample_sequential_chain(self, min_len: int = 3, max_len: int = 6) -> Dict:
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

        # [逻辑] 长度不足
        if len(path_nodes) < min_len:
            # 加大惩罚力度：如果这个起点总是导致短路，以后少选它
            self.node_starts[start_node] += 1.0
            return None

        # [逻辑] 暂不更新全局计数器
        # 我们把更新逻辑移到 generate_chains 里，只有确定收录了才更新
        # 但我们需要返回 edges_taken 以便外层更新
        return self._format_chain(path_nodes, edges_taken), edges_taken, start_node

    def _format_chain(self, nodes: List[str], edges: List[Tuple]) -> Dict:
        """格式化"""
        blueprint = {
            "pattern": "sequential",
            "tools_involved": [],
            "scenario_graph": [],
        }
        for node in nodes:
            attrs = self.graph.nodes[node]
            blueprint["tools_involved"].append({
                "name": node,
                "description": attrs.get("desc", ""),
                "category": attrs.get("category", "general"),
            })
        for i, (u, v, key) in enumerate(edges):
            edge_data = self.graph.get_edge_data(u, v)[key]
            blueprint["scenario_graph"].append({
                "step": i + 1,
                "from_tool": u,
                "to_tool": v,
                "dependency": {
                    "parameter": edge_data.get("parameter"),
                    "relation": "provides_input_for",
                },
            })
        return blueprint

    def generate_chains(
        self,
        count: int = 10,
        min_len: int = 3,
        max_len: int = 6,
        max_retries: int = 500,
    ) -> List[Dict]:
        """
        批量生成，带有去重和防卡死机制。
        Args:
            max_retries: 连续失败多少次后认为图已饱和，提前结束。
        """
        chains = []
        # [修改 1] 使用 Set 存储路径指纹 (StartNode -> ... -> EndNode)
        unique_hashes = set()

        # [修改 2] 连续失败计数器
        fail_streak = 0

        # 使用 tqdm 显示进度
        with tqdm(total=count, desc="Sampling Chains", unit="chain") as pbar:
            while len(chains) < count:
                # 尝试采样
                result = self.sample_sequential_chain(min_len=min_len, max_len=max_len)

                if result:
                    blueprint, edges_taken, start_node = result

                    # 生成指纹：将工具名连接起来
                    chain_signature = "->".join([
                        t["name"] for t in blueprint["tools_involved"]
                    ])
                    chain_hash = hashlib.md5(chain_signature.encode()).hexdigest()

                    if chain_hash not in unique_hashes:
                        # === 这是一个全新的有效链 ===
                        unique_hashes.add(chain_hash)
                        chains.append(blueprint)

                        # 更新全局计数器 (Decay 生效)
                        self.node_starts[start_node] += 1
                        for e in edges_taken:
                            self.edge_visits[e] += 1

                        # 重置失败计数，更新进度条
                        fail_streak = 0
                        pbar.update(1)
                    else:
                        # 重复了
                        fail_streak += 1
                else:
                    # 路径太短或死胡同
                    fail_streak += 1

                # [修改 3] 防卡死机制
                if fail_streak >= max_retries:
                    logger.warning(
                        f"Hit max retries ({max_retries}). Graph might be saturated for these constraints."
                    )
                    break

        stats = self.get_coverage_stats()
        logger.info(
            f"Generated {len(chains)} unique chains. Final Coverage: {stats['coverage_ratio']}"
        )
        return chains
