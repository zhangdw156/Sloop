"""
测试工具图谱构建器 (ToolGraphBuilder)

为 sloop/engine/graph.py 的核心功能编写单元测试。
"""

from unittest.mock import patch

import pytest  # 导入pytest以使用usefixtures

# import networkx as nx  # 可能有循环导入问题
from sloop.engine.graph import ToolGraphBuilder
from sloop.models.schema import ToolDefinition
from tests.utils import get_current_test_logger

# 获取当前测试文件的日志器
test_logger = get_current_test_logger()


@pytest.fixture
def patch_plt_savefig():
    """Fixture to patch plt.savefig"""
    with patch("sloop.engine.graph.plt.savefig"):
        yield


def get_mock_tools():
    """创建模拟工具数据"""
    return [
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
            description="Get menu for a restaurant using restaurant_id",
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
            description="Order food from menu using dish_id and restaurant_id",
            parameters={
                "type": "object",
                "properties": {
                    "dish_id": {"type": "string", "description": "Dish ID"},
                    "restaurant_id": {"type": "string", "description": "Restaurant ID"},
                },
                "required": ["dish_id"],
            },
        ),
        ToolDefinition(
            name="get_weather",
            description="Get weather information for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Location"}
                },
                "required": ["location"],
            },
        ),
    ]


def get_builder():
    """创建构建器实例"""
    mock_tools = get_mock_tools()
    builder = ToolGraphBuilder(mock_tools)
    builder.build()
    return builder


def test_initialization():
    """测试初始化"""
    test_logger.info("🔧 测试 ToolGraphBuilder 初始化")
    mock_tools = get_mock_tools()
    builder = ToolGraphBuilder(mock_tools)

    assert len(builder.tools) == 4
    assert len(builder.tool_map) == 4
    assert builder.graph is None  # 尚未构建

    test_logger.info("✅ 初始化测试通过")


def test_build_graph():
    """测试图构建"""
    test_logger.info("🏗️ 测试图构建")
    mock_tools = get_mock_tools()
    builder = ToolGraphBuilder(mock_tools)
    graph = builder.build()

    # 检查图的基本属性（不使用networkx直接导入）
    assert graph is not None
    assert hasattr(graph, "nodes")
    assert hasattr(graph, "edges")

    # 检查节点数量
    assert len(graph.nodes) == 4  # 4个工具

    # 检查所有节点都存在
    expected_nodes = {"find_restaurants", "get_menu", "order_food", "get_weather"}
    assert set(graph.nodes) == expected_nodes

    test_logger.info("✅ 图构建测试通过")


def test_dependency_analysis():
    """测试依赖关系分析"""
    test_logger.info("🔍 测试依赖关系分析")
    mock_tools = get_mock_tools()
    builder = ToolGraphBuilder(mock_tools)

    # find_restaurants -> get_menu (get_menu需要restaurant_id，find_restaurants返回restaurant_id)
    assert builder._has_dependency(
        mock_tools[0], mock_tools[1]
    )  # find_restaurants -> get_menu

    # find_restaurants -> order_food (order_food需要restaurant_id)
    assert builder._has_dependency(
        mock_tools[0], mock_tools[2]
    )  # find_restaurants -> order_food

    # get_menu -> order_food (order_food需要restaurant_id，get_menu使用restaurant_id)
    assert builder._has_dependency(
        mock_tools[1], mock_tools[2]
    )  # get_menu -> order_food

    # get_weather 不应该依赖其他工具
    assert not builder._has_dependency(mock_tools[3], mock_tools[0])
    assert not builder._has_dependency(mock_tools[3], mock_tools[1])
    assert not builder._has_dependency(mock_tools[3], mock_tools[2])

    test_logger.info("✅ 依赖分析测试通过")


def test_get_required_params():
    """测试获取必需参数"""
    test_logger.info("📋 测试获取必需参数")
    mock_tools = get_mock_tools()
    builder = ToolGraphBuilder(mock_tools)

    # find_restaurants 需要 city
    assert builder._get_required_params(mock_tools[0]) == ["city"]

    # get_menu 需要 restaurant_id
    assert builder._get_required_params(mock_tools[1]) == ["restaurant_id"]

    # order_food 需要 dish_id (restaurant_id是可选的)
    assert builder._get_required_params(mock_tools[2]) == ["dish_id"]

    # get_weather 需要 location
    assert builder._get_required_params(mock_tools[3]) == ["location"]

    test_logger.info("✅ 必需参数测试通过")


def test_sample_tool_chain():
    """测试工具链采样"""
    test_logger.info("🎲 测试工具链采样")
    builder = get_builder()
    chains = []

    # 采样多次以测试随机性
    for _i in range(10):
        chain = builder.sample_tool_chain(min_length=1, max_length=3)
        assert isinstance(chain, list)
        assert len(chain) >= 1
        assert len(chain) <= 3
        assert all(isinstance(tool, str) for tool in chain)
        chains.append(chain)

    # 检查至少有一些不同的链（证明随机性）
    unique_chains = {tuple(chain) for chain in chains}
    assert len(unique_chains) > 1

    test_logger.info("✅ 工具链采样测试通过")


def test_sample_tool_chain_edge_cases():
    """测试采样边界情况"""
    test_logger.info("🔄 测试采样边界情况")
    mock_tools = get_mock_tools()

    # 空工具列表
    empty_builder = ToolGraphBuilder([])
    empty_builder.build()
    chain = empty_builder.sample_tool_chain()
    assert chain == []

    # 单个工具
    single_builder = ToolGraphBuilder([mock_tools[0]])
    single_builder.build()
    chain = single_builder.sample_tool_chain(min_length=1, max_length=1)
    assert chain == ["find_restaurants"]

    test_logger.info("✅ 边界情况测试通过")


def test_get_tool_category():
    """测试获取工具类别"""
    test_logger.info("🏷️ 测试获取工具类别")
    builder = get_builder()

    # 没有category的工具返回"general"
    assert builder._get_tool_category("find_restaurants") == "general"
    assert builder._get_tool_category("nonexistent") == "general"

    test_logger.info("✅ 工具类别测试通过")


def test_calculate_domain_stickiness():
    """测试领域粘性计算"""
    test_logger.info("📌 测试领域粘性计算")
    builder = get_builder()

    # 同领域（都为general）
    assert builder._calculate_domain_stickiness("general", "find_restaurants") == 1.0

    # 不同领域
    assert builder._calculate_domain_stickiness("finance", "find_restaurants") == 0.3

    test_logger.info("✅ 领域粘性测试通过")


def test_are_related_categories():
    """测试类别相关性检查"""
    test_logger.info("🔗 测试类别相关性")
    builder = get_builder()

    # 相关类别
    assert builder._are_related_categories("finance", "business")
    assert builder._are_related_categories("food", "restaurant")

    # 不相关类别
    assert not builder._are_related_categories("finance", "music")

    test_logger.info("✅ 类别相关性测试通过")


def test_get_graph_stats():
    """测试图统计信息"""
    test_logger.info("📊 测试图统计")
    builder = get_builder()

    stats = builder.get_graph_stats()

    assert "nodes" in stats
    assert "edges" in stats
    assert "start_nodes" in stats
    assert "end_nodes" in stats

    assert stats["nodes"] == 4
    assert isinstance(stats["edges"], int)
    assert isinstance(stats["start_nodes"], int)
    assert isinstance(stats["end_nodes"], int)

    test_logger.info("✅ 图统计测试通过")


@pytest.mark.usefixtures("patch_plt_savefig")
def test_visualize_graph():
    """测试图可视化"""
    test_logger.info("📈 测试图可视化")
    builder = get_builder()

    # 应该成功保存（即使matplotlib未安装也会处理）
    builder.visualize_graph("test_graph.png")

    # 如果matplotlib可用，应该调用savefig
    # 这里不做严格检查，因为matplotlib可能未安装

    test_logger.info("✅ 图可视化测试通过")


# ==================== 集成测试 ====================


def run_integration_test():
    """运行集成测试"""
    test_logger.info("🔧 ToolGraphBuilder 集成测试")
    test_logger.info("=" * 50)

    # 创建测试工具
    test_tools = [
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
            description="Get menu for a restaurant using restaurant_id",
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
            description="Order food from menu using dish_id",
            parameters={
                "type": "object",
                "properties": {
                    "dish_id": {"type": "string", "description": "Dish ID"},
                },
                "required": ["dish_id"],
            },
        ),
    ]

    test_logger.info("📋 测试工具数据:")
    for tool in test_tools:
        test_logger.info(f"  - {tool.name}: {tool.description}")
    test_logger.info("")

    # 初始化构建器
    test_logger.info("🔧 初始化ToolGraphBuilder...")
    builder = ToolGraphBuilder(test_tools)
    builder.build()

    test_logger.info("📊 图统计:")
    stats = builder.get_graph_stats()
    test_logger.info(f"  节点数量: {stats['nodes']}")
    test_logger.info(f"  边数量: {stats['edges']}")
    test_logger.info(f"  起始节点: {stats['start_nodes']}")
    test_logger.info(f"  结束节点: {stats['end_nodes']}")
    test_logger.info("")

    # 采样工具链
    test_logger.info("🎲 采样工具链...")
    for i in range(3):
        chain = builder.sample_tool_chain(min_length=2, max_length=3)
        test_logger.info(f"  链 {i + 1}: {' -> '.join(chain)}")

    test_logger.info("")
    test_logger.info("✅ ToolGraphBuilder 集成测试完成！")


if __name__ == "__main__":
    # 运行集成测试
    run_integration_test()
