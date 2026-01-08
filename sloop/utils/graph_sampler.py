import random
import networkx as nx
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from sloop.utils.logger import logger

class GraphSampler:
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        # 覆盖率记忆模块
        self.edge_visits = defaultdict(int) 
        self.node_starts = defaultdict(int)
        self.total_edges = graph.number_of_edges()
        
    def reset_coverage(self):
        """重置所有访问记录"""
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
            "coverage_ratio": f"{coverage:.2%}"
        }

    def _select_start_nodes(self, k: int = 1) -> List[str]:
        """选择 k 个起始节点 (支持单点或多点)"""
        valid_starts = [n for n in self.graph.nodes() if self.graph.out_degree(n) > 0]
        if not valid_starts: return []
        
        # 加上随机扰动排序，优先选没去过的
        valid_starts.sort(key=lambda n: self.node_starts[n] + random.random())
        
        # 取前 k 个
        return valid_starts[:k]

    def _get_decayed_neighbors(self, node: str, top_n: int = 5) -> List[Tuple]:
        """
        获取邻居，根据访问次数衰减权重，返回 Top-N 候选
        Return: List[(next_node, edge_key, edge_attr, decayed_weight)]
        """
        successors = list(self.graph.successors(node))
        if not successors: return []
        
        candidates = []
        for succ in successors:
            edge_data = self.graph.get_edge_data(node, succ)
            # MultiGraph 可能有多条边，遍历所有边
            for key, attr in edge_data.items():
                w = attr.get('weight', 0.5)
                # 衰减公式：1 / (1 + 访问次数)
                decay = 1.0 / (1.0 + self.edge_visits[(node, succ, key)])
                candidates.append((succ, key, attr, w * decay))
        
        # 按衰减后的权重排序
        candidates.sort(key=lambda x: x[3], reverse=True)
        # 返回前 N 个
        return candidates[:top_n]

    def sample_dag(self, max_depth: int = 4, max_branch: int = 2, start_count: int = 1) -> Dict:
        """
        通用的采样引擎：支持 线性链、分叉树、并行多链
        """
        # 1. 初始化前沿 (Frontier)
        actual_starts = self._select_start_nodes(start_count)
        if not actual_starts: return None
        
        nodes_collected = set(actual_starts)
        edges_collected = [] # list of (u, v, key)
        
        # 当前活跃层
        current_layer = list(actual_starts)
        local_visited = set(actual_starts) # 防止本次采样出现环
        
        for _ in range(max_depth):
            next_layer = []
            
            for node in current_layer:
                # 获取候选邻居 (Top 5)
                candidates = self._get_decayed_neighbors(node, top_n=5)
                if not candidates: continue
                
                # 决定分叉数量: 线性模式(1), 分叉模式(1~max)
                num_branches = random.randint(1, min(len(candidates), max_branch))
                
                # 从候选里随机选 (加权随机更有趣，这里简单随机选 Top N 里的几个)
                chosen_branches = random.sample(candidates, num_branches)
                
                for next_node, key, _, _ in chosen_branches:
                    if next_node in local_visited:
                        continue 
                        
                    edges_collected.append((node, next_node, key))
                    nodes_collected.add(next_node)
                    local_visited.add(next_node)
                    next_layer.append(next_node)
            
            if not next_layer:
                break
            current_layer = next_layer
            
        # 过滤掉太短的图
        if len(edges_collected) == 0:
            return None
            
        # 更新全局计数器 (仅在确定采用后更新)
        for n in actual_starts: self.node_starts[n] += 1
        for u, v, k in edges_collected: self.edge_visits[(u, v, k)] += 1
            
        return self._format_blueprint(nodes_collected, edges_collected)

    def _format_blueprint(self, nodes: Set[str], edges: List[Tuple]) -> Dict:
        """通用格式化工具"""
        blueprint = {
            "pattern": "dag", # 统一叫 DAG，具体形态由数据决定
            "tools_involved": [],
            "scenario_graph": []
        }
        
        # 1. 工具列表
        for node_name in nodes:
            node_attrs = self.graph.nodes[node_name]
            blueprint["tools_involved"].append({
                "name": node_name,
                "description": node_attrs.get("desc", ""),
                "category": node_attrs.get("category", "")
            })
            
        # 2. 依赖关系
        for u, v, key in edges:
            edge_attr = self.graph.get_edge_data(u, v)[key]
            blueprint["scenario_graph"].append({
                "from_tool": u,
                "to_tool": v,
                "dependency": {
                    "parameter": edge_attr.get("parameter"),
                    "relation": "provides_input_for"
                }
            })
            
        return blueprint

    def generate_blueprints(self, count: int = 10) -> List[Dict]:
        """
        生成混合类型的蓝图
        通过调节 sample_dag 的参数，覆盖所有场景
        """
        blueprints = []
        attempts = 0
        max_attempts = count * 5
        
        while len(blueprints) < count and attempts < max_attempts:
            attempts += 1
            rand = random.random()
            bp = None
            
            if rand < 0.5:
                # 场景A: 纯线性链 (Step-by-Step)
                # 1个起点, 不分叉
                bp = self.sample_dag(max_depth=5, max_branch=1, start_count=1)
                if bp: bp["pattern"] = "sequential" # 标记一下方便人类看
                
            elif rand < 0.8:
                # 场景B: 分叉任务 (Decision Making)
                # 1个起点, 允许分叉
                bp = self.sample_dag(max_depth=3, max_branch=2, start_count=1)
                if bp: bp["pattern"] = "branching"
                
            else:
                # 场景C: 并行任务 (Multi-Intent)
                # 2个起点, 允许各自发展
                bp = self.sample_dag(max_depth=4, max_branch=1, start_count=2)
                if bp: bp["pattern"] = "parallel"
                
            if bp: blueprints.append(bp)
            
        # 打印覆盖率
        stats = self.get_coverage_stats()
        logger.info(f"Generated {len(blueprints)} blueprints. Coverage: {stats['coverage_ratio']}")
        return blueprints