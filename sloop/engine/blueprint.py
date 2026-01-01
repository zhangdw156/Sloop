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

    def __init__(self, tools: List[ToolDefinition], mode: str = "graph"):
        """
        初始化蓝图生成器

        参数:
            tools: 工具定义列表
            mode: 生成模式 ("graph" 或 "rag")
        """
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        self.mode = mode

        # 初始化工具图谱构建器
        self.graph_builder = ToolGraphBuilder(tools)
        self.graph_builder.build()

        # 获取并打印图谱统计信息
        stats = self.graph_builder.get_graph_stats()
        logger.info(f"📊 工具图谱构建完成:\n   - 节点数量: {stats['nodes']}\n   - 边数量: {stats['edges']}\n   - 起始节点 (入度为0): {stats['start_nodes']}\n   - 结束节点 (出度为0): {stats['end_nodes']}")

        # 初始化全局不放回采样状态
        self.all_start_nodes = self.graph_builder.get_start_nodes()
        if not self.all_start_nodes:
            # 如果没有入度为0的节点，使用所有节点
            self.all_start_nodes = list(self.tool_map.keys())
        self.used_start_nodes = set()

        logger.info(f"📋 发现 {len(self.all_start_nodes)} 个起始节点")

        # 初始化 RAG 相关组件（如果启用）
        if self.mode == "rag":
            from sloop.engine.rag import ToolRetrievalEngine
            from sloop.agents.selector import SelectorAgent

            logger.info("🔍 初始化 RAG 引擎...")
            self.rag_engine = ToolRetrievalEngine()
            self.rag_engine.build(tools)

            logger.info("🤖 初始化选择智能体...")
            self.selector_agent = SelectorAgent()

            logger.info("✅ RAG 模式初始化完成")
        else:
            self.rag_engine = None
            self.selector_agent = None

        logger.info(f"BlueprintGenerator initialized with {len(tools)} tools (mode: {mode})")

    def _select_diverse_start_node(self) -> str:
        """
        选择多样化的起始节点（全局不放回采样）

        实现全局不放回采样策略，确保在批量生成时优先遍历所有未使用的起始工具。

        返回:
            选中的起始节点名称
        """
        # 计算当前未使用的起始节点
        available = [node for node in self.all_start_nodes if node not in self.used_start_nodes]

        # 重置机制：如果所有节点都已使用，重置状态
        if not available:
            logger.info(f"🔄 重置起始节点使用状态 (已遍历 {len(self.used_start_nodes)} 个节点)")
            self.used_start_nodes.clear()
            available = self.all_start_nodes.copy()

        # 随机选择一个未使用的节点
        import random
        selected_node = random.choice(available)

        # 记录使用状态
        self.used_start_nodes.add(selected_node)

        logger.info(f"🎯 选择起始节点: {selected_node} (剩余未使用: {len(available) - 1})")
        return selected_node

    def _sample_rag_tool_chain(self, chain_length: int) -> List[str]:
        """
        使用 RAG 增强采样工具链

        参数:
            chain_length: 目标链长度

        返回:
            采样得到的工具链
        """
        logger.info(f"🎯 开始 RAG 增强采样 (目标长度: {chain_length})")

        # 1. 使用全局不放回采样选择起始工具
        current_tool_name = self._select_diverse_start_node()
        tool_chain = [current_tool_name]
        current_tool = self.tool_map[current_tool_name]

        # 2. 循环采样直到达到目标长度或决定结束
        while len(tool_chain) < chain_length:
            logger.info(f"🔄 当前链条: {' -> '.join(tool_chain)}")

            # 获取 Graph 邻居（显式候选）
            graph_neighbors = self.graph_builder.get_neighbors(current_tool_name)
            graph_candidates = [self.tool_map[name] for name in graph_neighbors if name in self.tool_map]

            # 获取 RAG 相似工具（隐式候选）
            rag_candidates = []
            if self.rag_engine:
                rag_names = self.rag_engine.search(current_tool, top_k=5)
                rag_candidates = [self.tool_map[name] for name in rag_names if name in self.tool_map and name not in graph_neighbors]

            # 合并候选，去重
            all_candidates = graph_candidates + rag_candidates
            # 排除已在链条中的工具
            available_candidates = [tool for tool in all_candidates if tool.name not in tool_chain]

            if not available_candidates:
                logger.info("⚠️ 没有更多可用候选，提前结束")
                break

            logger.info(f"📋 候选工具: {[t.name for t in available_candidates]}")

            # 调用 Selector 做决策
            selected_name = self.selector_agent.select_next_tool(tool_chain, available_candidates)

            if selected_name is None:
                logger.info("🏁 Selector 决定结束任务")
                break

            if selected_name not in self.tool_map:
                logger.warning(f"Selected tool {selected_name} not found, ending chain")
                break

            # 添加到链条
            tool_chain.append(selected_name)
            current_tool_name = selected_name
            current_tool = self.tool_map[current_tool_name]

            logger.info(f"✅ 选择工具: {selected_name}")

        logger.info(f"🎯 RAG 采样完成，最终链条: {' -> '.join(tool_chain)}")
        return tool_chain

    def _sample_graph_tool_chain(self, chain_length: int) -> List[str]:
        """
        使用图谱采样工具链（带全局不放回起始节点）

        参数:
            chain_length: 目标链长度

        返回:
            采样得到的工具链
        """
        logger.info(f"🎯 开始图谱采样 (目标长度: {chain_length})")

        # 1. 使用全局不放回采样选择起始工具
        current_tool_name = self._select_diverse_start_node()
        tool_chain = [current_tool_name]

        # 2. 使用图谱的领域粘性逻辑继续采样
        remaining_length = chain_length - 1
        if remaining_length > 0:
            # 获取图谱采样的后续链
            extended_chain = self.graph_builder.sample_tool_chain(
                min_length=remaining_length,
                max_length=remaining_length
            )
            if extended_chain and len(extended_chain) > 1:
                # 跳过第一个元素（因为我们已经选择了起始节点）
                tool_chain.extend(extended_chain[1:])

        logger.info(f"🎯 图谱采样完成，最终链条: {' -> '.join(tool_chain)}")
        return tool_chain

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

                # 1. 采样工具链
                if self.mode == "rag":
                    tool_chain = self._sample_rag_tool_chain(chain_length)
                else:
                    tool_chain = self._sample_graph_tool_chain(chain_length)

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
