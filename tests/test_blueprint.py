"""
测试蓝图生成器 (BlueprintGenerator)

迁移自 sloop/engine/blueprint.py 的测试代码，并添加单元测试。
"""

import json

# 自定义logger，用于测试日志记录
import logging
import os
from unittest.mock import patch

# import pytest  # 注释掉pytest，使用标准unittest
from sloop.engine.blueprint import BlueprintGenerator
from sloop.models.blueprint import Blueprint
from sloop.models.schema import ToolDefinition

# 创建logs目录（如果不存在）
test_log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(test_log_dir, exist_ok=True)

# 配置测试logger
test_logger = logging.getLogger("test_blueprint")
test_logger.setLevel(logging.DEBUG)

# 文件handler
log_file = os.path.join(test_log_dir, "test_blueprint.log")
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


def test_initialization():
    """测试初始化"""
    test_logger.info("🔧 测试 BlueprintGenerator 初始化")
    mock_tools = get_mock_tools()
    generator = BlueprintGenerator(mock_tools)

    assert len(generator.tools) == 3
    assert len(generator.tool_map) == 3
    assert generator.graph_builder is not None

    # 检查图谱统计
    stats = generator.graph_builder.get_graph_stats()
    assert stats["nodes"] == 3
    assert stats["edges"] >= 0

    test_logger.info("✅ 初始化测试通过")


@patch("sloop.engine.blueprint.chat_completion")
def test_generate_success(mock_chat):
    """测试成功生成蓝图"""
    test_logger.info("🎯 测试蓝图生成成功场景")

    # 模拟LLM响应
    mock_response = json.dumps({
        "intent": "查找餐厅并点餐",
        "valid": True,
        "initial_state": {"restaurant_found": False, "menu_loaded": False},
        "expected_state": {"restaurant_found": True, "menu_loaded": True},
    })
    mock_chat.return_value = mock_response

    mock_tools = get_mock_tools()
    generator = BlueprintGenerator(mock_tools)
    blueprint = generator.generate(chain_length=2, max_retries=1)

    assert isinstance(blueprint, Blueprint)
    assert blueprint.intent == "查找餐厅并点餐"
    assert len(blueprint.required_tools) == 2
    assert blueprint.initial_state == {"restaurant_found": False, "menu_loaded": False}
    assert blueprint.expected_state == {"restaurant_found": True, "menu_loaded": True}

    test_logger.info("✅ 成功生成测试通过")


@patch("sloop.engine.blueprint.chat_completion")
def test_generate_with_invalid_response(mock_chat):
    """测试处理无效LLM响应"""
    test_logger.info("❌ 测试处理无效LLM响应")

    # 模拟无效JSON响应
    mock_chat.return_value = "invalid json response"

    mock_tools = get_mock_tools()
    generator = BlueprintGenerator(mock_tools)
    blueprint = generator.generate(chain_length=2, max_retries=2)

    # 应该返回后备蓝图
    assert isinstance(blueprint, Blueprint)
    assert "执行工具链" in blueprint.intent

    test_logger.info("✅ 无效响应处理测试通过")


@patch("sloop.engine.blueprint.chat_completion")
def test_generate_with_llm_error(mock_chat):
    """测试处理LLM调用错误"""
    test_logger.info("🚨 测试处理LLM调用错误")

    # 模拟LLM调用失败
    mock_chat.return_value = "调用错误: connection timeout"

    mock_tools = get_mock_tools()
    generator = BlueprintGenerator(mock_tools)
    blueprint = generator.generate(chain_length=2, max_retries=1)

    # 应该返回后备蓝图
    assert isinstance(blueprint, Blueprint)
    assert "执行工具链" in blueprint.intent

    test_logger.info("✅ LLM错误处理测试通过")


def test_generate_multiple():
    """测试批量生成蓝图"""
    test_logger.info("📊 测试批量生成蓝图")

    with patch("sloop.engine.blueprint.chat_completion") as mock_chat:
        mock_response = json.dumps({
            "intent": "测试意图",
            "valid": True,
            "initial_state": {},
            "expected_state": {},
        })
        mock_chat.return_value = mock_response

        mock_tools = get_mock_tools()
        generator = BlueprintGenerator(mock_tools)
        blueprints = generator.generate_multiple(count=3, chain_length=2)

        assert len(blueprints) == 3
        for bp in blueprints:
            assert isinstance(bp, Blueprint)

    test_logger.info("✅ 批量生成测试通过")


def test_validate_blueprint_data():
    """测试蓝图数据验证"""
    test_logger.info("🔍 测试蓝图数据验证")

    mock_tools = get_mock_tools()
    generator = BlueprintGenerator(mock_tools)

    # 测试有效数据
    valid_data = {
        "intent": "测试意图",
        "initial_state": {"key": "value"},
        "expected_state": {"result": True},
    }
    expected_chain = ["tool1", "tool2"]

    validated = generator._validate_blueprint_data(valid_data, expected_chain)

    assert validated["intent"] == "测试意图"
    assert validated["required_tools"] == expected_chain
    assert validated["ground_truth"] == expected_chain
    assert validated["initial_state"] == {"key": "value"}
    assert validated["expected_state"] == {"result": True}

    test_logger.info("✅ 数据验证测试通过")


def test_fallback_blueprint_generation():
    """测试后备蓝图生成"""
    test_logger.info("🔧 测试后备蓝图生成")

    mock_tools = get_mock_tools()
    generator = BlueprintGenerator(mock_tools)
    tool_chain = ["find_restaurants", "get_menu"]

    fallback_bp = generator._generate_fallback_blueprint(tool_chain)

    assert isinstance(fallback_bp, Blueprint)
    assert "执行工具链" in fallback_bp.intent
    assert fallback_bp.required_tools == tool_chain
    assert fallback_bp.ground_truth == tool_chain
    assert isinstance(fallback_bp.initial_state, dict)
    assert isinstance(fallback_bp.expected_state, dict)

    test_logger.info("✅ 后备蓝图测试通过")


# ==================== 集成测试（迁移自原main方法） ====================


def run_integration_test():
    """运行集成测试（原main方法逻辑）"""
    test_logger.info("🔧 Blueprint Generator 集成测试")
    test_logger.info("=" * 50)

    # 创建模拟工具数据
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

    test_logger.info("📋 模拟工具数据:")
    for tool in mock_tools:
        test_logger.info(f"  - {tool.name}: {tool.description}")
    test_logger.info("")

    # 初始化生成器
    test_logger.info("🔧 初始化BlueprintGenerator...")
    generator = BlueprintGenerator(mock_tools)

    test_logger.info("📊 图谱统计:")
    stats = generator.graph_builder.get_graph_stats()
    test_logger.info(f"  节点数量: {stats['nodes']}")
    test_logger.info(f"  边数量: {stats['edges']}")
    test_logger.info("")

    # 生成蓝图（使用mock避免实际LLM调用）
    test_logger.info("🎯 生成Blueprint...")
    try:
        with patch("sloop.engine.blueprint.chat_completion") as mock_chat:
            mock_response = json.dumps({
                "intent": "查找餐厅并点餐",
                "valid": True,
                "initial_state": {"restaurant_found": False, "menu_loaded": False},
                "expected_state": {"restaurant_found": True, "menu_loaded": True},
            })
            mock_chat.return_value = mock_response

            blueprint = generator.generate(chain_length=2)

        test_logger.info("✅ 生成成功！")
        test_logger.info("\n📋 Blueprint详情:")
        test_logger.info(f"  意图: {blueprint.intent}")
        test_logger.info(f"  必需工具: {blueprint.required_tools}")
        test_logger.info(f"  真实工具链: {blueprint.ground_truth}")
        test_logger.info(f"  初始状态: {blueprint.initial_state}")
        test_logger.info(f"  期望状态: {blueprint.expected_state}")

        test_logger.info("\n📄 完整JSON:")
        test_logger.info(blueprint.model_dump_json(indent=2))

    except Exception as e:
        test_logger.error(f"❌ 生成失败: {e}")

        # 如果LLM调用失败，提供模拟结果
        test_logger.info("\n🔧 提供模拟Blueprint作为示例:")
        mock_blueprint = Blueprint(
            intent="查找餐厅并点餐",
            required_tools=["find_restaurants", "get_menu"],
            ground_truth=["find_restaurants", "get_menu"],
            initial_state={"restaurant_found": False, "menu_loaded": False},
            expected_state={"restaurant_found": True, "menu_loaded": True},
        )
        test_logger.info(mock_blueprint.model_dump_json(indent=2))

    test_logger.info("\n✅ Blueprint Generator 集成测试完成！")


if __name__ == "__main__":
    # 运行集成测试
    run_integration_test()
