"""
蓝图生成器 (Blueprint Generator)

连接工具图谱和LLM想象力，自动生成合理的对话蓝图。
"""

import json
import logging
from typing import List, Optional
from ..models import Blueprint, ToolDefinition
from .graph import ToolGraphBuilder
from ..utils.template import render_planner_prompt
from ..utils.llm import chat_completion

logger = logging.getLogger(__name__)


class BlueprintGenerator:
    """
    蓝图生成器

    基于工具图谱采样和LLM推理，自动生成对话蓝图。
    """

    def __init__(self, tools: List[ToolDefinition]):
        """
        初始化蓝图生成器

        参数:
            tools: 工具定义列表
        """
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}

        # 初始化工具图谱构建器
        self.graph_builder = ToolGraphBuilder(tools)
        self.graph_builder.build()

        logger.info(f"BlueprintGenerator initialized with {len(tools)} tools")

    def generate(self, chain_length: int = 3) -> Blueprint:
        """
        生成对话蓝图

        参数:
            chain_length: 工具链长度

        返回:
            生成的对话蓝图
        """
        logger.info(f"Generating blueprint with chain length {chain_length}")

        # 1. 从图谱中采样工具链
        tool_chain = self.graph_builder.sample_tool_chain(
            min_length=max(1, chain_length - 1),
            max_length=chain_length
        )

        if not tool_chain:
            raise ValueError("无法采样到有效的工具链")

        logger.info(f"Sampled tool chain: {tool_chain}")

        # 2. 获取工具定义
        tool_definitions = []
        for tool_name in tool_chain:
            if tool_name in self.tool_map:
                tool_definitions.append(self.tool_map[tool_name])
            else:
                logger.warning(f"Tool {tool_name} not found in tool map")

        if not tool_definitions:
            raise ValueError("没有找到有效的工具定义")

        # 3. 构造和发送提示
        prompt = render_planner_prompt(tool_chain, tool_definitions)

        logger.info("Sending prompt to LLM for blueprint generation")

        # 4. 调用LLM生成蓝图
        llm_response = chat_completion(
            prompt=prompt,
            system_message="You are an expert AI dataset generator. Always respond with valid JSON.",
            json_mode=True
        )

        if not llm_response or llm_response.startswith("调用错误"):
            raise RuntimeError(f"LLM调用失败: {llm_response}")

        # 5. 解析和验证响应
        try:
            blueprint_data = json.loads(llm_response)
            logger.info("Successfully parsed LLM response")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {llm_response}")
            raise ValueError(f"LLM响应不是有效的JSON: {e}")

        # 6. 验证和修正数据
        validated_data = self._validate_blueprint_data(blueprint_data, tool_chain)

        # 7. 创建Blueprint对象
        blueprint = Blueprint(**validated_data)

        logger.info(f"Successfully generated blueprint: {blueprint.intent}")
        return blueprint

    def _validate_blueprint_data(self, data: dict, expected_chain: List[str]) -> dict:
        """
        验证和修正蓝图数据

        参数:
            data: LLM返回的原始数据
            expected_chain: 期望的工具链

        返回:
            验证后的数据字典
        """
        validated = {}

        # 验证intent
        if "intent" not in data or not isinstance(data["intent"], str):
            raise ValueError("缺少有效的intent字段")
        validated["intent"] = data["intent"].strip()

        # 验证required_tools（可以是采样的链或LLM建议的链）
        if "required_tools" in data and isinstance(data["required_tools"], list):
            validated["required_tools"] = data["required_tools"]
        else:
            validated["required_tools"] = expected_chain

        # 强制设置ground_truth为采样的链
        validated["ground_truth"] = expected_chain

        # 验证initial_state
        if "initial_state" not in data or not isinstance(data["initial_state"], dict):
            logger.warning("缺少initial_state，使用默认值")
            validated["initial_state"] = {}
        else:
            validated["initial_state"] = data["initial_state"]

        # 验证expected_state
        if "expected_state" not in data or not isinstance(data["expected_state"], dict):
            logger.warning("缺少expected_state，使用默认值")
            validated["expected_state"] = {}
        else:
            validated["expected_state"] = data["expected_state"]

        return validated

    def generate_multiple(self, count: int = 5, chain_length: int = 3) -> List[Blueprint]:
        """
        生成多个蓝图

        参数:
            count: 生成数量
            chain_length: 工具链长度

        返回:
            蓝图列表
        """
        blueprints = []
        for i in range(count):
            try:
                blueprint = self.generate(chain_length)
                blueprints.append(blueprint)
                logger.info(f"Generated blueprint {i+1}/{count}: {blueprint.intent}")
            except Exception as e:
                logger.error(f"Failed to generate blueprint {i+1}: {e}")
                continue

        return blueprints


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("🔧 Blueprint Generator 测试")
    print("=" * 50)

    # 创建模拟工具数据
    mock_tools = [
        ToolDefinition(
            name="find_restaurants",
            description="Find restaurants and return restaurant_id",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        ),
        ToolDefinition(
            name="get_menu",
            description="Get menu for a restaurant",
            parameters={
                "type": "object",
                "properties": {
                    "restaurant_id": {"type": "string", "description": "Restaurant ID"}
                },
                "required": ["restaurant_id"]
            }
        ),
        ToolDefinition(
            name="order_food",
            description="Order food from menu",
            parameters={
                "type": "object",
                "properties": {
                    "dish_id": {"type": "string", "description": "Dish ID"},
                    "restaurant_id": {"type": "string", "description": "Restaurant ID"}
                },
                "required": ["dish_id"]
            }
        )
    ]

    print("📋 模拟工具数据:")
    for tool in mock_tools:
        print(f"  - {tool.name}: {tool.description}")
    print()

    # 初始化生成器
    print("🔧 初始化BlueprintGenerator...")
    generator = BlueprintGenerator(mock_tools)

    print("📊 图谱统计:")
    stats = generator.graph_builder.get_graph_stats()
    print(f"  节点数量: {stats['nodes']}")
    print(f"  边数量: {stats['edges']}")
    print()

    # 生成蓝图
    print("🎯 生成Blueprint...")
    try:
        blueprint = generator.generate(chain_length=2)

        print("✅ 生成成功！")
        print("\n📋 Blueprint详情:")
        print(f"  意图: {blueprint.intent}")
        print(f"  必需工具: {blueprint.required_tools}")
        print(f"  真实工具链: {blueprint.ground_truth}")
        print(f"  初始状态: {blueprint.initial_state}")
        print(f"  期望状态: {blueprint.expected_state}")

        print("\n📄 完整JSON:")
        print(blueprint.model_dump_json(indent=2))

    except Exception as e:
        print(f"❌ 生成失败: {e}")

        # 如果LLM调用失败，提供模拟结果
        print("\n🔧 提供模拟Blueprint作为示例:")
        mock_blueprint = Blueprint(
            intent="查找餐厅并点餐",
            required_tools=["find_restaurants", "get_menu"],
            ground_truth=["find_restaurants", "get_menu"],
            initial_state={"restaurant_found": False, "menu_loaded": False},
            expected_state={"restaurant_found": True, "menu_loaded": True}
        )
        print(mock_blueprint.model_dump_json(indent=2))

    print("\n✅ Blueprint Generator 测试完成！")
