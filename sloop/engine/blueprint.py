"""
蓝图生成器 (Blueprint Generator)

连接工具图谱和LLM想象力，自动生成合理的对话蓝图。
"""

import json
import logging
from typing import List, Optional
from sloop.models import Blueprint, ToolDefinition
from sloop.engine.graph import ToolGraphBuilder
from sloop.utils.template import render_planner_prompt
from sloop.utils.llm import chat_completion

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

    def generate(self, chain_length: int = 3, max_retries: int = 3) -> Blueprint:
        """
        生成对话蓝图，包含合理性验证和重试机制

        参数:
            chain_length: 工具链长度
            max_retries: 最大重试次数

        返回:
            生成的对话蓝图
        """
        logger.info(f"Generating blueprint with chain length {chain_length}, max_retries {max_retries}")

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries}")

                # 1. 从图谱中采样工具链
                tool_chain = self.graph_builder.sample_tool_chain(
                    min_length=max(1, chain_length - 1),
                    max_length=chain_length
                )

                if not tool_chain:
                    logger.warning(f"Attempt {attempt + 1}: Failed to sample tool chain, retrying...")
                    continue

                logger.info(f"Sampled tool chain: {tool_chain}")

                # 2. 获取工具定义
                tool_definitions = []
                for tool_name in tool_chain:
                    if tool_name in self.tool_map:
                        tool_definitions.append(self.tool_map[tool_name])
                    else:
                        logger.warning(f"Tool {tool_name} not found in tool map")

                if not tool_definitions:
                    logger.warning(f"Attempt {attempt + 1}: No valid tool definitions found, retrying...")
                    continue

                # 3. 构造和发送提示
                prompt = render_planner_prompt(tool_chain, tool_definitions)

                logger.info("Sending prompt to LLM for blueprint generation")

                # 4. 调用LLM生成蓝图
                llm_response = chat_completion(
                    prompt=prompt,
                    system_message="你是专家级AI数据集生成器。始终用有效的JSON格式响应。",
                    json_mode=True
                )

                if not llm_response or llm_response.startswith("调用错误"):
                    logger.warning(f"Attempt {attempt + 1}: LLM call failed: {llm_response}, retrying...")
                    continue

                # 5. 解析和验证响应
                try:
                    blueprint_data = json.loads(llm_response)
                    logger.info("Successfully parsed LLM response")
                except json.JSONDecodeError as e:
                    logger.warning(f"Attempt {attempt + 1}: Failed to parse LLM response as JSON: {llm_response}, retrying...")
                    continue

                # 6. 检查蓝图合理性
                if not blueprint_data.get("valid", True):
                    reason = blueprint_data.get("reason", "Unknown reason")
                    logger.warning(f"Attempt {attempt + 1}: Blueprint marked as invalid: {reason}, retrying...")
                    continue

                # 7. 验证和修正数据
                validated_data = self._validate_blueprint_data(blueprint_data, tool_chain)

                # 8. 创建Blueprint对象
                blueprint = Blueprint(**validated_data)

                logger.info(f"Successfully generated valid blueprint: {blueprint.intent}")
                return blueprint

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")
                continue

        # 所有重试都失败了，返回一个简单的默认蓝图
        logger.error(f"All {max_retries} attempts failed, generating fallback blueprint")
        return self._generate_fallback_blueprint(tool_chain)

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

    def _generate_fallback_blueprint(self, tool_chain: List[str]) -> Blueprint:
        """
        生成后备蓝图，当所有重试都失败时使用

        参数:
            tool_chain: 工具链列表

        返回:
            简单的后备蓝图
        """
        logger.info("Generating fallback blueprint")

        # 构建简单的intent
        tool_names = [name for name in tool_chain]  # 直接使用工具名
        intent = f"执行工具链: {' -> '.join(tool_names)}"

        # 简单的状态
        initial_state = {f"{name}_executed": False for name in tool_chain}
        expected_state = {f"{name}_executed": True for name in tool_chain}

        return Blueprint(
            intent=intent,
            required_tools=tool_chain,
            ground_truth=tool_chain,
            initial_state=initial_state,
            expected_state=expected_state
        )

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
    logger.info("🔧 Blueprint Generator 测试")
    logger.info("=" * 50)

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

    logger.info("📋 模拟工具数据:")
    for tool in mock_tools:
        logger.info(f"  - {tool.name}: {tool.description}")
    logger.info("")

    # 初始化生成器
    logger.info("🔧 初始化BlueprintGenerator...")
    generator = BlueprintGenerator(mock_tools)

    logger.info("📊 图谱统计:")
    stats = generator.graph_builder.get_graph_stats()
    logger.info(f"  节点数量: {stats['nodes']}")
    logger.info(f"  边数量: {stats['edges']}")
    logger.info("")

    # 生成蓝图
    logger.info("🎯 生成Blueprint...")
    try:
        blueprint = generator.generate(chain_length=2)

        logger.info("✅ 生成成功！")
        logger.info("\n📋 Blueprint详情:")
        logger.info(f"  意图: {blueprint.intent}")
        logger.info(f"  必需工具: {blueprint.required_tools}")
        logger.info(f"  真实工具链: {blueprint.ground_truth}")
        logger.info(f"  初始状态: {blueprint.initial_state}")
        logger.info(f"  期望状态: {blueprint.expected_state}")

        logger.info("\n📄 完整JSON:")
        logger.info(blueprint.model_dump_json(indent=2))

    except Exception as e:
        logger.error(f"❌ 生成失败: {e}")

        # 如果LLM调用失败，提供模拟结果
        logger.info("\n🔧 提供模拟Blueprint作为示例:")
        mock_blueprint = Blueprint(
            intent="查找餐厅并点餐",
            required_tools=["find_restaurants", "get_menu"],
            ground_truth=["find_restaurants", "get_menu"],
            initial_state={"restaurant_found": False, "menu_loaded": False},
            expected_state={"restaurant_found": True, "menu_loaded": True}
        )
        logger.info(mock_blueprint.model_dump_json(indent=2))

    logger.info("\n✅ Blueprint Generator 测试完成！")
