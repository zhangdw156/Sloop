"""
蓝图生成器 (Blueprint Generator)

连接工具图谱和LLM想象力，自动生成合理的对话蓝图。
"""

import json
from typing import List

from sloop.engine.graph import ToolGraphBuilder
from sloop.models import Blueprint, ToolDefinition
from sloop.utils.llm import chat_completion
from sloop.utils.logger import logger
from sloop.utils.template import render_planner_prompt


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
        logger.info(
            f"Generating blueprint with chain length {chain_length}, max_retries {max_retries}"
        )

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries}")

                # 1. 从图谱中采样工具链
                tool_chain = self.graph_builder.sample_tool_chain(
                    min_length=max(1, chain_length - 1), max_length=chain_length
                )

                if not tool_chain:
                    logger.warning(
                        f"Attempt {attempt + 1}: Failed to sample tool chain, retrying..."
                    )
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
                    logger.warning(
                        f"Attempt {attempt + 1}: No valid tool definitions found, retrying..."
                    )
                    continue

                # 3. 构造和发送提示
                prompt = render_planner_prompt(tool_chain, tool_definitions)

                logger.info("Sending prompt to LLM for blueprint generation")

                # 4. 调用LLM生成蓝图
                llm_response = chat_completion(
                    prompt=prompt,
                    system_message="",
                    json_mode=True,
                )

                if not llm_response or llm_response.startswith("调用错误"):
                    logger.warning(
                        f"Attempt {attempt + 1}: LLM call failed: {llm_response}, retrying..."
                    )
                    continue

                # 5. 解析和验证响应
                try:
                    blueprint_data = json.loads(llm_response)
                    logger.info("Successfully parsed LLM response")
                except json.JSONDecodeError:
                    logger.warning(
                        f"Attempt {attempt + 1}: Failed to parse LLM response as JSON: {llm_response}, retrying..."
                    )
                    continue

                # 6. 检查蓝图合理性
                if not blueprint_data.get("valid", True):
                    reason = blueprint_data.get("reason", "Unknown reason")
                    logger.warning(
                        f"Attempt {attempt + 1}: Blueprint marked as invalid: {reason}, retrying..."
                    )
                    continue

                # 7. 验证和修正数据
                validated_data = self._validate_blueprint_data(
                    blueprint_data, tool_chain
                )

                # 8. 创建Blueprint对象
                blueprint = Blueprint(**validated_data)

                logger.info(
                    f"Successfully generated valid blueprint: {blueprint.intent}"
                )
                return blueprint

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")
                continue

        # 所有重试都失败了，返回一个简单的默认蓝图
        logger.error(
            f"All {max_retries} attempts failed, generating fallback blueprint"
        )
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

        # 强制设置required_tools和ground_truth为采样的链
        validated["required_tools"] = expected_chain
        validated["ground_truth"] = expected_chain

        # 验证initial_state
        if "initial_state" not in data or not isinstance(data["initial_state"], dict):
            logger.warning("缺少initial_state，使用默认值")
            validated["initial_state"] = {}
        else:
            validated["initial_state"] = data["initial_state"]

        # 验证expected_state，确保键值对足够简单
        if "expected_state" not in data or not isinstance(data["expected_state"], dict):
            logger.warning("缺少expected_state，使用默认值")
            validated["expected_state"] = {}
        else:
            # 简化expected_state，只保留布尔值和简单类型
            simplified_state = {}
            for key, value in data["expected_state"].items():
                if (
                    isinstance(value, bool)
                    or isinstance(value, (str, int, float))
                    and len(str(value)) < 50
                ):
                    simplified_state[key] = value
                else:
                    logger.warning(f"简化expected_state: 跳过复杂值 {key}: {value}")
            validated["expected_state"] = simplified_state

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
        tool_names = list(tool_chain)  # 直接使用工具名
        intent = f"执行工具链: {' -> '.join(tool_names)}"

        # 简单的状态
        initial_state = {f"{name}_executed": False for name in tool_chain}
        expected_state = {f"{name}_executed": True for name in tool_chain}

        return Blueprint(
            intent=intent,
            required_tools=tool_chain,
            ground_truth=tool_chain,
            initial_state=initial_state,
            expected_state=expected_state,
        )

    def generate_multiple(
        self, count: int = 5, chain_length: int = 3
    ) -> List[Blueprint]:
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
                logger.info(f"Generated blueprint {i + 1}/{count}: {blueprint.intent}")
            except Exception as e:
                logger.error(f"Failed to generate blueprint {i + 1}: {e}")
                continue

        return blueprints
