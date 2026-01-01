"""
测试 SelectorAgent

测试裁判智能体的工具选择功能。
"""

import logging
import os

import pytest

from sloop.agents.selector import SelectorAgent
from sloop.models import ToolDefinition

# 创建logs目录（如果不存在）
test_log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(test_log_dir, exist_ok=True)

# 配置测试logger
test_logger = logging.getLogger("test_selector")
test_logger.setLevel(logging.DEBUG)

# 文件handler
log_file = os.path.join(test_log_dir, "test_selector.log")
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)

# 控制台handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)

# 添加handlers
test_logger.addHandler(file_handler)
test_logger.addHandler(console_handler)


class TestSelectorAgent:
    """SelectorAgent 测试类"""

    @pytest.fixture
    def selector_agent(self):
        """创建 SelectorAgent 实例"""
        return SelectorAgent()

    @pytest.fixture
    def mock_candidates(self):
        """创建模拟候选工具"""
        return [
            ToolDefinition(
                name="recommend_clothes",
                description="根据天气推荐穿衣",
                parameters={
                    "type": "object",
                    "properties": {
                        "weather": {"type": "string", "description": "天气情况"},
                    },
                    "required": ["weather"],
                },
            ),
            ToolDefinition(
                name="book_flight",
                description="预订机票",
                parameters={
                    "type": "object",
                    "properties": {
                        "destination": {"type": "string", "description": "目的地"},
                        "date": {"type": "string", "description": "出发日期"},
                    },
                    "required": ["destination"],
                },
            ),
            ToolDefinition(
                name="delete_database",
                description="删除数据库",
                parameters={
                    "type": "object",
                    "properties": {
                        "database_name": {"type": "string", "description": "数据库名称"},
                    },
                    "required": ["database_name"],
                },
            ),
        ]

    def test_init(self, selector_agent):
        """测试初始化"""
        assert selector_agent is not None
        assert hasattr(selector_agent, 'settings')

    def test_select_next_tool_empty_candidates(self, selector_agent):
        """测试空候选列表"""
        result = selector_agent.select_next_tool([], [])
        assert result is None

    def test_select_next_tool_with_chain(self, selector_agent, mock_candidates):
        """测试有链条的情况"""
        # 这里主要测试方法能正常调用，具体返回值依赖 LLM
        # 在实际测试中可能需要 mock LLM 调用
        result = selector_agent.select_next_tool(["get_weather"], mock_candidates)
        # 结果可能是工具名或 None，取决于 LLM 响应
        assert result is None or isinstance(result, str)

    def test_select_next_tool_no_chain(self, selector_agent, mock_candidates):
        """测试无链条的情况"""
        result = selector_agent.select_next_tool([], mock_candidates)
        assert result is None or isinstance(result, str)

    def test_select_next_tool_with_flight_chain(self, selector_agent, mock_candidates):
        """测试预订机票后的情况"""
        result = selector_agent.select_next_tool(["book_flight"], mock_candidates)
        assert result is None or isinstance(result, str)
