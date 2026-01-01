"""
测试 ToolRetrievalEngine

测试向量检索引擎的功能。
"""

import tempfile
from pathlib import Path

import pytest

from sloop.engine.rag import ToolRetrievalEngine
from sloop.models import ToolDefinition
from tests.utils import get_current_test_logger

# 获取当前测试文件的日志器
test_logger = get_current_test_logger()


class TestToolRetrievalEngine:
    """ToolRetrievalEngine 测试类"""

    @pytest.fixture
    def temp_cache_dir(self):
        """创建临时缓存目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def mock_tools(self):
        """创建模拟工具"""
        return [
            ToolDefinition(
                name="get_weather",
                description="获取指定城市的天气信息",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"},
                        "date": {"type": "string", "description": "日期"},
                    },
                    "required": ["city"],
                },
            ),
            ToolDefinition(
                name="search_restaurants",
                description="搜索指定城市的餐厅",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"},
                        "cuisine": {"type": "string", "description": "菜系类型"},
                        "price_range": {"type": "string", "description": "价格范围"},
                    },
                    "required": ["city"],
                },
            ),
            ToolDefinition(
                name="book_hotel",
                description="预订酒店房间",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"},
                        "check_in": {"type": "string", "description": "入住日期"},
                        "check_out": {"type": "string", "description": "退房日期"},
                        "guests": {"type": "integer", "description": "入住人数"},
                    },
                    "required": ["city", "check_in", "check_out"],
                },
            ),
            ToolDefinition(
                name="send_email",
                description="发送电子邮件",
                parameters={
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "description": "收件人邮箱"},
                        "subject": {"type": "string", "description": "邮件主题"},
                        "body": {"type": "string", "description": "邮件正文"},
                    },
                    "required": ["to", "subject", "body"],
                },
            ),
            ToolDefinition(
                name="calculate_distance",
                description="计算两地之间的距离",
                parameters={
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string", "description": "起点"},
                        "destination": {"type": "string", "description": "终点"},
                        "mode": {"type": "string", "description": "出行方式", "enum": ["driving", "walking", "transit"]},
                    },
                    "required": ["origin", "destination"],
                },
            ),
        ]

    def test_init(self, temp_cache_dir):
        """测试初始化"""
        engine = ToolRetrievalEngine(cache_dir=str(temp_cache_dir))
        assert engine.cache_dir == temp_cache_dir
        assert engine.index is None
        assert engine.tool_names == []

    def test_build_and_search(self, temp_cache_dir, mock_tools):
        """测试构建索引和搜索"""
        engine = ToolRetrievalEngine(cache_dir=str(temp_cache_dir))

        # 构建索引
        engine.build(mock_tools, force=True)

        # 检查文件是否创建
        assert engine.index_path.exists()
        assert engine.names_path.exists()

        # 检查索引是否加载
        assert engine.index is not None
        assert len(engine.tool_names) == len(mock_tools)

        # 测试搜索
        query_tool = mock_tools[0]  # get_weather
        results = engine.search(query_tool, top_k=3)

        # 结果应该是工具名称列表
        assert isinstance(results, list)
        assert len(results) <= 3
        for result in results:
            assert result in [tool.name for tool in mock_tools]

    def test_search_without_index(self, temp_cache_dir):
        """测试在没有索引时搜索"""
        engine = ToolRetrievalEngine(cache_dir=str(temp_cache_dir))

        query_tool = ToolDefinition(
            name="test_tool",
            description="测试工具",
            parameters={"type": "object", "properties": {}, "required": []},
        )

        results = engine.search(query_tool, top_k=5)
        assert results == []

    def test_build_idempotent(self, temp_cache_dir, mock_tools):
        """测试重复构建的幂等性"""
        engine = ToolRetrievalEngine(cache_dir=str(temp_cache_dir))

        # 第一次构建
        engine.build(mock_tools, force=True)
        first_names = engine.tool_names.copy()

        # 第二次构建（不强制）
        engine.build(mock_tools, force=False)
        second_names = engine.tool_names.copy()

        # 应该保持不变
        assert first_names == second_names

    def test_build_force_rebuild(self, temp_cache_dir, mock_tools):
        """测试强制重建"""
        engine = ToolRetrievalEngine(cache_dir=str(temp_cache_dir))

        # 第一次构建
        engine.build(mock_tools[:3], force=True)
        assert len(engine.tool_names) == 3

        # 强制重建所有工具
        engine.build(mock_tools, force=True)
        assert len(engine.tool_names) == len(mock_tools)


def run_integration_test():
    """运行集成测试（原rag.py main方法逻辑）"""
    test_logger.info("🔍 ToolRetrievalEngine 集成测试")
    test_logger.info("=" * 50)

    # 创建测试工具（使用temp目录）
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        engine = ToolRetrievalEngine(cache_dir=temp_dir)

        mock_tools = [
            ToolDefinition(
                name="get_weather",
                description="获取指定城市的天气信息",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"},
                    },
                    "required": ["city"],
                },
            ),
            ToolDefinition(
                name="search_restaurants",
                description="搜索指定城市的餐厅",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"},
                    },
                    "required": ["city"],
                },
            ),
        ]

        # 构建索引
        test_logger.info("🏗️ 构建索引...")
        engine.build(mock_tools, force=True)

        # 测试搜索
        test_logger.info("🔍 测试搜索...")
        query_tool = mock_tools[0]  # get_weather
        results = engine.search(query_tool, top_k=3)
        test_logger.info(f"🎯 相似工具: {results}")

        test_logger.info("✅ 集成测试完成！")


if __name__ == "__main__":
    # 运行集成测试
    run_integration_test()
