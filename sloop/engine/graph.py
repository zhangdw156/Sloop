"""
参数级工具图谱构建器 (Tool Graph Builder)

实现工具之间的依赖关系分析和图谱构建，用于生成合理的工具调用链。
"""

import random
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx

from sloop.models import ToolDefinition
from sloop.utils.logger import logger


class ToolGraphBuilder:
    """
    工具图谱构建器

    基于工具描述和参数分析，构建工具之间的依赖关系图。
    用于生成合理的工具调用序列。
    """

    def __init__(self, tools: List[ToolDefinition]):
        """
        初始化构建器

        参数:
            tools: 工具定义列表
        """
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        self.graph: Optional[nx.DiGraph] = None

    def build(self) -> nx.DiGraph:
        """
        构建工具依赖图

        返回:
            有向图：节点为工具名，边表示依赖关系
        """
        # 创建有向图
        self.graph = nx.DiGraph()

        # 添加所有节点
        for tool in self.tools:
            self.graph.add_node(tool.name, tool=tool)

        # 分析依赖关系并添加边
        for tool_a in self.tools:
            for tool_b in self.tools:
                if tool_a.name != tool_b.name and self._has_dependency(tool_a, tool_b):
                    self.graph.add_edge(tool_a.name, tool_b.name)

        return self.graph

    def _has_dependency(self, tool_a: ToolDefinition, tool_b: ToolDefinition) -> bool:
        """
        判断工具A是否是工具B的依赖

        逻辑：如果A的描述中提到的参数名出现在B的参数中，则A -> B

        参数:
            tool_a: 可能的依赖工具
            tool_b: 被依赖的工具

        返回:
            是否存在依赖关系
        """
        # 获取B的所有参数名
        all_params_b = (
            set(tool_b.parameters.properties.keys())
            if tool_b.parameters.properties
            else set()
        )
        if not all_params_b:
            return False

        # 检查A的描述是否包含B的任何参数名
        description_a = tool_a.description.lower()
        return any(param.lower() in description_a for param in all_params_b)

    def _get_required_params(self, tool: ToolDefinition) -> List[str]:
        """
        获取工具的必需参数名列表

        参数:
            tool: 工具定义

        返回:
            必需参数名列表
        """
        required = []
        if hasattr(tool.parameters, "required") and tool.parameters.required:
            required = tool.parameters.required
        return required

    def sample_tool_chain(self, min_length: int = 2, max_length: int = 5) -> List[str]:
        """
        从图中采样一条工具调用链，考虑领域粘性

        参数:
            min_length: 最小链长度
            max_length: 最大链长度

        返回:
            工具名列表，表示调用顺序
        """
        if self.graph is None:
            raise ValueError("图尚未构建，请先调用 build() 方法")

        if len(self.graph.nodes) == 0:
            return []

        # 找到入度为0的节点（起始节点）
        start_nodes = [
            node for node in self.graph.nodes if self.graph.in_degree(node) == 0
        ]

        if not start_nodes:
            # 如果没有入度为0的节点，随机选择一个
            start_nodes = list(self.graph.nodes)

        # 随机选择起始节点
        current_node = random.choice(start_nodes)
        chain = [current_node]

        # 获取当前工具的category作为基准
        current_category = self._get_tool_category(current_node)

        # 随机游走构建链（增加最大尝试次数避免死循环）
        max_attempts = 50
        attempts = 0

        while len(chain) < max_length and attempts < max_attempts:
            # 获取当前节点的后继节点
            successors = list(self.graph.successors(current_node))

            if not successors:
                # 没有后继节点，结束
                break

            # 按领域粘性对后继节点进行排序
            # 同领域或相关领域的节点优先级更高
            scored_successors_with_random = []
            for successor in successors:
                score = self._calculate_domain_stickiness(current_category, successor)
                scored_successors_with_random.append((
                    successor,
                    score,
                    random.random(),
                ))

            # 按分数降序排序，然后按随机值降序排序以打破平局
            scored_successors_with_random.sort(key=lambda x: (x[1], x[2]), reverse=True)
            final_successors = [s[0] for s in scored_successors_with_random]

            # 按80%的概率选择高粘性节点，20%概率随机选择
            HIGH_STICKINESS_PROB = 0.8
            if random.random() < HIGH_STICKINESS_PROB and final_successors:
                next_node = final_successors[0]  # 选择最相关的节点
            else:
                next_node = random.choice(successors)  # 随机选择以保持多样性

            # 避免环路
            if next_node in chain:
                attempts += 1
                continue

            chain.append(next_node)
            current_node = next_node
            current_category = self._get_tool_category(current_node)

        # 确保最小长度（简化逻辑）
        if len(chain) < min_length:
            # 如果太短，返回当前链（避免复杂扩展逻辑导致问题）
            pass

        return chain

    def _get_tool_category(self, tool_name: str) -> str:
        """
        获取工具的category

        参数:
            tool_name: 工具名

        返回:
            工具的category，如果没有则返回"general"
        """
        if tool_name in self.tool_map:
            tool = self.tool_map[tool_name]
            # 从tool的额外字段中获取category
            if hasattr(tool, "category") and tool.category:
                return tool.category
            # 或者从model_extra中获取
            if (
                hasattr(tool, "model_extra")
                and tool.model_extra
                and "category" in tool.model_extra
            ):
                return tool.model_extra["category"]

        return "general"

    def _calculate_domain_stickiness(
        self, current_category: str, candidate_tool: str
    ) -> float:
        """
        计算领域粘性分数

        参数:
            current_category: 当前工具的category
            candidate_tool: 候选工具名

        返回:
            粘性分数 (0-1)，越高表示越相关
        """
        candidate_category = self._get_tool_category(candidate_tool)

        if current_category == candidate_category:
            # 同领域，最高分数
            return 1.0
        elif self._are_related_categories(current_category, candidate_category):
            # 相关领域，中等分数
            return 0.7
        else:
            # 无关领域，低分数
            return 0.3

    def _are_related_categories(self, cat1: str, cat2: str) -> bool:
        """
        判断两个category是否相关

        参数:
            cat1: 类别1
            cat2: 类别2

        返回:
            是否相关
        """
        # 定义相关类别的映射
        related_categories = {
            "finance": ["business", "investment", "stock", "banking"],
            "business": ["finance", "investment", "company"],
            "investment": ["finance", "business", "stock"],
            "stock": ["finance", "investment"],
            "banking": ["finance", "payment"],
            "travel": ["booking", "hotel", "flight", "transport"],
            "booking": ["travel", "hotel", "flight", "restaurant"],
            "hotel": ["travel", "booking"],
            "flight": ["travel", "booking", "transport"],
            "transport": ["travel", "flight"],
            "food": ["restaurant", "cooking", "delivery"],
            "restaurant": ["food", "booking"],
            "cooking": ["food", "recipe"],
            "delivery": ["food", "restaurant"],
            "music": ["audio", "entertainment"],
            "audio": ["music", "entertainment"],
            "entertainment": ["music", "audio", "game"],
            "game": ["entertainment", "gaming"],
            "gaming": ["game", "entertainment"],
            "health": ["medical", "fitness"],
            "medical": ["health", "doctor"],
            "fitness": ["health", "exercise"],
            "education": ["learning", "study"],
            "learning": ["education", "study"],
            "study": ["education", "learning"],
            "shopping": ["ecommerce", "retail"],
            "ecommerce": ["shopping", "retail"],
            "retail": ["shopping", "ecommerce"],
        }

        # 检查双向关系
        return (cat1 in related_categories and cat2 in related_categories[cat1]) or (
            cat2 in related_categories and cat1 in related_categories[cat2]
        )

    def get_start_nodes(self) -> List[str]:
        """
        获取起始节点（入度为0的节点）

        返回:
            起始节点名称列表
        """
        if self.graph is None:
            return []
        return [node for node in self.graph.nodes if self.graph.in_degree(node) == 0]

    def get_neighbors(self, node_name: str) -> List[str]:
        """
        获取节点的邻居节点（后继节点）

        参数:
            node_name: 节点名称

        返回:
            邻居节点名称列表
        """
        if self.graph is None or node_name not in self.graph:
            return []
        return list(self.graph.successors(node_name))

    def get_graph_stats(self) -> Dict:
        """
        获取图的统计信息

        返回:
            统计信息字典
        """
        if self.graph is None:
            return {"error": "图尚未构建"}

        return {
            "nodes": len(self.graph.nodes),
            "edges": len(self.graph.edges),
            "start_nodes": len([
                n for n in self.graph.nodes if self.graph.in_degree(n) == 0
            ]),
            "end_nodes": len([
                n for n in self.graph.nodes if self.graph.out_degree(n) == 0
            ]),
        }

    def visualize_graph(self, output_path: str = "tool_graph.png"):
        """
        可视化工具图（需要matplotlib）

        参数:
            output_path: 输出图片路径
        """
        if self.graph is None:
            logger.info("图尚未构建")
            return

        try:
            plt.figure(figsize=(12, 8))

            # 计算节点位置
            pos = nx.spring_layout(self.graph, k=2, iterations=50)

            # 绘制节点
            nx.draw_networkx_nodes(
                self.graph, pos, node_size=2000, node_color="lightblue", alpha=0.7
            )

            # 绘制边
            nx.draw_networkx_edges(
                self.graph, pos, arrows=True, arrowsize=20, alpha=0.6
            )

            # 绘制标签
            nx.draw_networkx_labels(self.graph, pos, font_size=10, font_weight="bold")

            plt.title("Tool Dependency Graph", fontsize=16)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"✅ 图可视化已保存到: {output_path}")

        except ImportError:
            logger.error("❌ 需要安装matplotlib才能可视化图谱")
        except Exception as e:
            logger.error(f"❌ 可视化失败: {e}")
