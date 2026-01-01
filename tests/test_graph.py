"""
测试 ToolGraphBuilder 功能
"""

import os

import pytest

from sloop.engine import ToolGraphBuilder
from sloop.models import ToolDefinition


class TestToolGraphBuilder:
    """测试工具图谱构建器"""

    @pytest.fixture
    def mock_tools(self):
        """创建模拟工具数据"""
        return [
            ToolDefinition(
                name="find_restaurants",
                description="Find restaurants and return restaurant_id",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"],
                },
            ),
            ToolDefinition(
                name="get_menu",
                description="Get menu for a restaurant",
                parameters={
                    "type": "object",
                    "properties": {
                        "restaurant_id": {
                            "type": "string",
                            "description": "Restaurant ID",
                        }
                    },
                    "required": ["restaurant_id"],
                },
            ),
            ToolDefinition(
                name="order_food",
                description="Order food from menu",
                parameters={
                    "type": "object",
                    "properties": {
                        "dish_id": {"type": "string", "description": "Dish ID"},
                        "restaurant_id": {
                            "type": "string",
                            "description": "Restaurant ID",
                        },
                    },
                    "required": ["dish_id"],
                },
            ),
        ]

    def test_build_graph(self, mock_tools):
        """测试图构建功能"""
        builder = ToolGraphBuilder(mock_tools)
        graph = builder.build()

        # 验证图的基本属性
        assert len(graph.nodes) == 3
        assert len(graph.edges) > 0  # 应该有依赖关系

        # 验证节点
        assert "find_restaurants" in graph.nodes
        assert "get_menu" in graph.nodes
        assert "order_food" in graph.nodes

    def test_dependency_detection(self, mock_tools):
        """测试依赖关系检测"""
        builder = ToolGraphBuilder(mock_tools)
        graph = builder.build()

        # find_restaurants -> get_menu (因为find_restaurants返回restaurant_id，get_menu需要restaurant_id)
        assert ("find_restaurants", "get_menu") in graph.edges

        # 验证统计信息
        stats = builder.get_graph_stats()
        assert stats["nodes"] == 3
        assert stats["edges"] >= 1

    def test_sample_tool_chain(self, mock_tools):
        """测试工具链采样"""
        builder = ToolGraphBuilder(mock_tools)
        builder.build()

        # 测试采样
        chain = builder.sample_tool_chain(min_length=1, max_length=3)

        # 验证链的基本属性
        assert isinstance(chain, list)
        assert len(chain) >= 1
        assert len(chain) <= 3

        # 验证所有工具名都在原始工具中
        tool_names = [tool.name for tool in mock_tools]
        for tool_name in chain:
            assert tool_name in tool_names

    def test_empty_tools(self):
        """测试空工具列表"""
        builder = ToolGraphBuilder([])
        graph = builder.build()

        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

        chain = builder.sample_tool_chain()
        assert chain == []

    def test_single_tool(self):
        """测试单个工具"""
        tools = [
            ToolDefinition(
                name="single_tool",
                description="A single tool",
                parameters={
                    "type": "object",
                    "properties": {"param": {"type": "string"}},
                    "required": ["param"],
                },
            )
        ]

        builder = ToolGraphBuilder(tools)
        graph = builder.build()

        assert len(graph.nodes) == 1
        assert len(graph.edges) == 0

        chain = builder.sample_tool_chain(min_length=1, max_length=2)
        assert chain == ["single_tool"]

    def test_graph_stats(self, mock_tools):
        """测试图统计功能"""
        builder = ToolGraphBuilder(mock_tools)
        builder.build()

        stats = builder.get_graph_stats()

        assert "nodes" in stats
        assert "edges" in stats
        assert "start_nodes" in stats
        assert "end_nodes" in stats

        assert stats["nodes"] == 3
        assert stats["start_nodes"] >= 1  # 至少有一个起始节点
        assert stats["end_nodes"] >= 1  # 至少有一个结束节点


def test_real_data_file_exists():
    """测试真实数据文件存在性（不加载内容）"""
    real_data_path = "tests/data/tools.json"
    if os.path.exists(real_data_path):
        # 只检查文件大小，不加载内容
        file_size = os.path.getsize(real_data_path)
        assert file_size > 0
        print(f"✅ 检测到真实数据文件，大小: {file_size:,} bytes")
    else:
        pytest.skip("真实数据文件不存在，跳过测试")


if __name__ == "__main__":
    # 手动运行测试
    print("🔧 Tool Graph Builder 手动测试")
    print("=" * 50)

    # 创建模拟数据
    mock_tools = [
        ToolDefinition(
            name="find_restaurants",
            description="Find restaurants and return restaurant_id",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        ),
        ToolDefinition(
            name="get_menu",
            description="Get menu for a restaurant",
            parameters={
                "type": "object",
                "properties": {
                    "restaurant_id": {"type": "string", "description": "Restaurant ID"}
                },
                "required": ["restaurant_id"],
            },
        ),
        ToolDefinition(
            name="order_food",
            description="Order food from menu",
            parameters={
                "type": "object",
                "properties": {
                    "dish_id": {"type": "string", "description": "Dish ID"},
                    "restaurant_id": {"type": "string", "description": "Restaurant ID"},
                },
                "required": ["dish_id"],
            },
        ),
    ]

    print("📋 模拟工具数据:")
    for tool in mock_tools:
        print(f"  - {tool.name}: {tool.description}")
    print()

    # 构建图谱
    builder = ToolGraphBuilder(mock_tools)
    graph = builder.build()

    print("📊 图构建结果:")
    stats = builder.get_graph_stats()
    print(f"  节点数量: {stats['nodes']}")
    print(f"  边数量: {stats['edges']}")
    print(f"  起始节点: {stats['start_nodes']}")
    print(f"  结束节点: {stats['end_nodes']}")
    print()

    # 显示边
    print("🔗 依赖关系 (Edges):")
    for edge in graph.edges:
        print(f"  {edge[0]} -> {edge[1]}")
    print()

    # 采样工具链
    print("🎲 随机采样工具链:")
    for i in range(3):
        chain = builder.sample_tool_chain(min_length=2, max_length=4)
        print(f"  链 {i + 1}: {' -> '.join(chain) if chain else '无'}")
    print()

    # 可选：真实数据统计（简化版，避免加载大文件）
    real_data_path = "tests/data/tools.json"
    if os.path.exists(real_data_path):
        print("📂 检测到真实数据文件...")
        try:
            # 只获取文件大小，不加载内容
            file_size = os.path.getsize(real_data_path)
            print(f"  文件大小: {file_size:,} bytes")
            print("  ℹ️ 为避免性能问题，跳过详细分析")

        except Exception as e:
            print(f"❌ 文件检查失败: {e}")
    else:
        print("ℹ️ 未找到真实数据文件，跳过统计分析")

    print("\n✅ Tool Graph Builder 测试完成！")
