import json
import os
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Set
from pathlib import Path

from sloop.models import ToolDefinition, ToolParameters
from sloop.utils.logger import logger

class ToolGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.tools: Dict[str, ToolDefinition] = {}

    def load_from_jsonl(self, file_path: str):
        """
        解析特定格式的 JSONL 文件（移除数量限制，加载所有条目）。
        格式特点：tools 字段是一个字符串化的 JSON list。
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Loading tools from {file_path} (no item limit)...")
        
        # 仅保留有效行数统计，无数量限制
        processed_lines = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # 解析 JSON 行
                    data = json.loads(line)
                    
                    # 处理两种格式：
                    # 1. 直接的工具定义 {"type": "function", "function": {...}}
                    # 2. 包含字符串化 tools 列表的格式 {"tools": "[{...}]"}

                    tools_list = []

                    # 检查是否是包含字符串化 tools 列表的格式
                    if "tools" in data and isinstance(data["tools"], str):
                        try:
                            tools_list = json.loads(data["tools"])
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse tools string: {e}")
                            continue
                    else:
                        # 直接将单个工具作为列表处理
                        tools_list = [data]

                    # 处理每个工具定义
                    for tool_dict in tools_list:
                        # 适配 OpenAI 格式: {"type": "function", "function": {...}}
                        if "function" in tool_dict:
                            func_data = tool_dict["function"]
                        else:
                            func_data = tool_dict

                        # 转换为 Pydantic 模型
                        tool = ToolDefinition(
                            name=func_data.get("name"),
                            description=func_data.get("description", ""),
                            parameters=ToolParameters(**func_data.get("parameters", {}))
                        )
                        
                        # 去重：如果同名工具已存在，跳过
                        if tool.name and tool.name not in self.tools:
                            self.tools[tool.name] = tool
                    
                    processed_lines += 1  # 统计处理的行数
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
                except Exception as e:
                    logger.error(f"Error processing line: {e}")

        logger.info(f"Loaded {len(self.tools)} unique tools from {processed_lines} valid records (all lines processed).")

    def build(self):
        """构建依赖图"""
        logger.info("Building dependency graph...")
        
        # 1. 添加节点
        for name, tool in self.tools.items():
            self.graph.add_node(name, desc=tool.description)

        # 2. 添加边 (基于参数依赖)
        # 逻辑：如果 Tool A 的输出/描述 包含 Tool B 的输入参数名，则可能存在 A -> B 的流转
        for name_a, tool_a in self.tools.items():
            for name_b, tool_b in self.tools.items():
                if name_a == name_b:
                    continue
                
                if self._has_dependency(tool_a, tool_b):
                    self.graph.add_edge(name_a, name_b)

        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")
        return self.graph

    def _has_dependency(self, producer: ToolDefinition, consumer: ToolDefinition) -> bool:
        """
        判断是否存在 Producer -> Consumer 的依赖。
        简化逻辑：检查 Producer 的描述是否覆盖了 Consumer 的必需参数。
        """
        consumer_required = set(consumer.parameters.required)
        if not consumer_required:
            return False

        # 检查 Producer 的描述文本 (转小写)
        producer_text = producer.description.lower()
        
        # 统计 Consumer 的必需参数在 Producer 描述中出现的比例
        matched_params = [p for p in consumer_required if p.lower() in producer_text]
        
        # 阈值：如果覆盖了至少一个必需参数，建立弱连接
        return len(matched_params) > 0

    def visualize(self, output_path: str = "tool_graph_vis.png"):
        """绘制并保存图谱"""
        if self.graph.number_of_nodes() == 0:
            logger.warning("Graph is empty, nothing to visualize.")
            return

        plt.figure(figsize=(15, 10))
        # 使用 shell layout 往往能更好地展示层次结构，或者 spring layout
        pos = nx.spring_layout(self.graph, k=0.5, iterations=50)
        
        # 绘制节点
        nx.draw_networkx_nodes(self.graph, pos, node_size=1000, node_color='skyblue', alpha=0.8)
        
        # 绘制边
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', arrows=True, alpha=0.5)
        
        # 绘制标签
        nx.draw_networkx_labels(self.graph, pos, font_size=8, font_family="sans-serif")

        plt.title(f"Tool Dependency Graph ({len(self.tools)} tools)")
        plt.axis('off')
        
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graph visualization saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
        finally:
            plt.close()

    def export_graph_json(self, output_path: str = "tool_graph.json"):
        """导出图谱结构为 JSON"""
        data = nx.node_link_data(self.graph)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Graph data exported to {output_path}")
