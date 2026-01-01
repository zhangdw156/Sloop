"""
测试 Blueprint Generator 功能
"""

import json
from unittest.mock import patch

import pytest

from sloop.engine import BlueprintGenerator
from sloop.models import Blueprint, ToolDefinition
from sloop.utils.template import render_planner_prompt


class TestBlueprintGenerator:
    """测试蓝图生成器"""

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

    def test_initialization(self, mock_tools):
        """测试生成器初始化"""
        generator = BlueprintGenerator(mock_tools)

        assert len(generator.tools) == 3
        assert len(generator.tool_map) == 3
        assert generator.graph_builder is not None

        # 检查图谱是否已构建
        stats = generator.graph_builder.get_graph_stats()
        assert stats["nodes"] == 3

    @patch("sloop.engine.blueprint.chat_completion")
    @patch("sloop.engine.graph.ToolGraphBuilder.sample_tool_chain")
    def test_generate_success(
        self, mock_sample_chain, mock_chat_completion, mock_tools
    ):
        """测试成功生成蓝图"""
        # 模拟固定的采样结果
        mock_sample_chain.return_value = ["find_restaurants", "get_menu"]

        # 模拟LLM响应
        mock_response = json.dumps({
            "intent": "查找餐厅并查看菜单",
            "required_tools": ["find_restaurants", "get_menu"],
            "ground_truth": ["find_restaurants", "get_menu"],
            "initial_state": {"restaurant_found": False},
            "expected_state": {"restaurant_found": True, "menu_loaded": True},
        })
        mock_chat_completion.return_value = mock_response

        generator = BlueprintGenerator(mock_tools)
        blueprint = generator.generate(chain_length=2)

        assert isinstance(blueprint, Blueprint)
        assert blueprint.intent == "查找餐厅并查看菜单"
        assert blueprint.ground_truth == ["find_restaurants", "get_menu"]
        assert blueprint.initial_state == {"restaurant_found": False}
        assert blueprint.expected_state == {
            "restaurant_found": True,
            "menu_loaded": True,
        }

    @patch("sloop.engine.blueprint.chat_completion")
    @patch("sloop.engine.graph.ToolGraphBuilder.sample_tool_chain")
    def test_generate_with_missing_fields(
        self, mock_sample_chain, mock_chat_completion, mock_tools
    ):
        """测试处理LLM响应缺少字段的情况"""
        # 模拟固定的采样结果
        mock_sample_chain.return_value = ["find_restaurants", "get_menu"]

        # 模拟不完整的LLM响应
        mock_response = json.dumps({
            "intent": "测试意图"
            # 缺少其他必需字段
        })
        mock_chat_completion.return_value = mock_response

        generator = BlueprintGenerator(mock_tools)
        blueprint = generator.generate(chain_length=2)

        # 应该使用默认值或采样的链
        assert isinstance(blueprint, Blueprint)
        assert blueprint.intent == "测试意图"
        assert blueprint.ground_truth == ["find_restaurants", "get_menu"]  # 采样结果
        assert blueprint.initial_state == {}  # 默认值
        assert blueprint.expected_state == {}  # 默认值

    @patch("sloop.engine.blueprint.chat_completion")
    def test_generate_llm_error(self, mock_chat_completion, mock_tools):
        """测试LLM调用失败的情况"""
        mock_chat_completion.return_value = "调用错误: 网络问题"

        generator = BlueprintGenerator(mock_tools)

        with pytest.raises(RuntimeError, match="LLM调用失败"):
            generator.generate(chain_length=2)

    @patch("sloop.engine.blueprint.chat_completion")
    def test_generate_invalid_json(self, mock_chat_completion, mock_tools):
        """测试LLM返回无效JSON的情况"""
        mock_chat_completion.return_value = "这不是有效的JSON响应"

        generator = BlueprintGenerator(mock_tools)

        with pytest.raises(ValueError, match="LLM响应不是有效的JSON"):
            generator.generate(chain_length=2)

    def test_validate_blueprint_data(self, mock_tools):
        """测试蓝图数据验证"""
        generator = BlueprintGenerator(mock_tools)

        # 测试有效数据
        valid_data = {
            "intent": "测试意图",
            "required_tools": ["tool1", "tool2"],
            "ground_truth": ["tool1", "tool2"],
            "initial_state": {"key": "value"},
            "expected_state": {"key": "new_value"},
        }

        result = generator._validate_blueprint_data(valid_data, ["tool1", "tool2"])
        assert result["intent"] == "测试意图"
        assert result["ground_truth"] == ["tool1", "tool2"]  # 强制使用期望链

    def test_validate_blueprint_data_missing_fields(self, mock_tools):
        """测试验证缺少字段的数据"""
        generator = BlueprintGenerator(mock_tools)

        # 测试缺少字段的数据
        incomplete_data = {
            "intent": "测试意图"
            # 缺少其他字段
        }

        result = generator._validate_blueprint_data(
            incomplete_data, ["find_restaurants", "get_menu"]
        )
        assert result["intent"] == "测试意图"
        assert result["ground_truth"] == ["find_restaurants", "get_menu"]  # 使用期望链
        assert result["initial_state"] == {}  # 默认值
        assert result["expected_state"] == {}  # 默认值

    def test_validate_blueprint_data_invalid_intent(self, mock_tools):
        """测试验证无效意图的数据"""
        generator = BlueprintGenerator(mock_tools)

        # 测试无效意图
        invalid_data = {
            "intent": 123,  # 应该是字符串
            "ground_truth": ["find_restaurants"],
        }

        with pytest.raises(ValueError, match="缺少有效的intent字段"):
            generator._validate_blueprint_data(invalid_data, ["find_restaurants"])

    def test_generate_empty_chain(self):
        """测试空工具列表的情况"""
        generator = BlueprintGenerator([])

        with pytest.raises(ValueError, match="无法采样到有效的工具链"):
            generator.generate(chain_length=2)

    @patch("sloop.engine.blueprint.chat_completion")
    def test_generate_multiple(self, mock_chat_completion, mock_tools):
        """测试批量生成蓝图"""
        # 模拟LLM响应
        mock_response = json.dumps({
            "intent": "批量测试意图",
            "required_tools": ["find_restaurants"],
            "ground_truth": ["find_restaurants"],
            "initial_state": {},
            "expected_state": {"done": True},
        })
        mock_chat_completion.return_value = mock_response

        generator = BlueprintGenerator(mock_tools)
        blueprints = generator.generate_multiple(count=2, chain_length=1)

        assert len(blueprints) == 2
        assert all(isinstance(bp, Blueprint) for bp in blueprints)
        assert all(bp.intent == "批量测试意图" for bp in blueprints)


@pytest.fixture
def mock_tools_global():
    """全局模拟工具数据fixture"""
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


def test_planner_prompt_rendering(mock_tools_global):
    """测试规划器提示模板渲染"""
    tool_chain = ["find_restaurants", "get_menu"]
    tool_definitions = mock_tools_global[:2]  # 只用前两个工具

    prompt = render_planner_prompt(tool_chain, tool_definitions)

    # 检查提示内容
    assert "find_restaurants" in prompt
    assert "get_menu" in prompt
    assert "Target Tool Chain" in prompt
    assert "Tool Definitions" in prompt
    assert "Output strictly in JSON format" in prompt
