"""
助手模拟器 (Assistant Agent)

模拟被测试的助手模型，根据工具定义和对话历史决定下一步行动。
"""

import json
import re
from typing import List

from sloop.models import ChatMessage, ToolCall, ToolDefinition
from sloop.utils.llm import chat_completion
from sloop.utils.logger import logger
from sloop.utils.template import (
    render_assistant_decide_prompt,
    render_assistant_reply_prompt,
    render_assistant_think_prompt,
    render_tool_call_gen_prompt,
)


class AssistantAgent:
    """
    助手智能体

    负责模拟被测试的助手模型，根据工具定义和对话历史生成响应，
    可能包含工具调用。
    """

    def __init__(self, tools: List[ToolDefinition]):
        """
        初始化助手智能体

        参数:
            tools: 可用的工具定义列表
        """
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}

        logger.info(f"AssistantAgent initialized with {len(tools)} tools")

    def parse_tool_calls(self, response: str) -> List[ToolCall]:
        """
        从响应中解析工具调用

        参数:
            response: 助手响应字符串

        返回:
            解析出的工具调用列表
        """
        tool_calls = []

        # 尝试解析JSON格式的工具调用
        # 查找类似 {"tool_name": "...", "arguments": {...}} 的模式（兼容name字段）
        json_pattern = r'\{[^}]*"(?:tool_name|name)"\s*:\s*"([^"]+)"[^}]*"arguments"\s*:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})\}'
        matches = re.findall(json_pattern, response, re.DOTALL)

        for match in matches:
            tool_name, args_str = match
            try:
                arguments = json.loads(args_str)
                if tool_name in self.tool_map:
                    tool_call = ToolCall(name=tool_name, arguments=arguments)
                    tool_calls.append(tool_call)
                    logger.info(f"Parsed tool call: {tool_name}")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool call arguments: {args_str}")

        # 如果没找到JSON格式，尝试查找函数调用模式
        if not tool_calls:
            # 查找 function_call 模式
            func_pattern = r'"function_call"\s*:\s*\{[^}]*"name"\s*:\s*"([^"]+)"[^}]*"arguments"\s*:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})\}'
            func_matches = re.findall(func_pattern, response, re.DOTALL)

            for match in func_matches:
                tool_name, args_str = match
                try:
                    arguments = json.loads(args_str)
                    if tool_name in self.tool_map:
                        tool_call = ToolCall(name=tool_name, arguments=arguments)
                        tool_calls.append(tool_call)
                        logger.info(f"Parsed function call: {tool_name}")
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to parse function call arguments: {args_str}"
                    )

        return tool_calls

    def should_call_tools(self, response: str) -> bool:
        """
        判断响应是否包含工具调用

        参数:
            response: 助手响应字符串

        返回:
            是否包含工具调用
        """
        return len(self.parse_tool_calls(response)) > 0

    def generate_thought(
        self, conversation_history: List[ChatMessage], context_hint: str = ""
    ) -> str:
        """
        生成助手思考过程 (Chain of Thought)

        参数:
            conversation_history: 对话历史消息列表
            context_hint: 栈上下文提示信息（可选）

        返回:
            思考过程字符串
        """
        logger.info("Generating assistant thought process (CoT)")

        # 使用模板渲染提示
        prompt = render_assistant_think_prompt(conversation_history, context_hint)

        # 调用LLM生成思考过程
        thought = chat_completion(
            prompt=prompt,
            system_message="",
            json_mode=False,
        )

        if not thought or thought.startswith("调用错误"):
            logger.error(f"Failed to generate thought: {thought}")
            return "我需要分析用户的请求并确定最佳响应方式。"

        return thought.strip()

    def decide_tool_use(self, thought: str) -> bool:
        """
        基于思考过程决定是否需要使用工具

        参数:
            thought: 思考过程字符串

        返回:
            是否需要使用工具
        """
        logger.info("Deciding whether to use tools based on thought process")

        # 使用模板渲染提示
        prompt = render_assistant_decide_prompt(thought, self.tools)

        decision = (
            chat_completion(
                prompt=prompt,
                system_message="",
                json_mode=False,
            )
            .strip()
            .upper()
        )

        needs_tools = decision.startswith("YES")
        logger.info(f"Tool use decision: {needs_tools}")
        return needs_tools

    def generate_tool_calls(
        self, thought: str, tools: List[ToolDefinition]
    ) -> List[ToolCall]:
        """
        基于思考过程生成工具调用

        参数:
            thought: 思考过程字符串
            tools: 可用的工具列表

        返回:
            工具调用列表
        """
        logger.info("Generating tool calls based on thought process")

        # 使用模板渲染提示
        prompt = render_tool_call_gen_prompt(thought, tools)

        response = chat_completion(
            prompt=prompt,
            system_message="",
            json_mode=True,
        )

        try:
            tool_calls_data = json.loads(response)
            if not isinstance(tool_calls_data, list):
                tool_calls_data = [tool_calls_data]

            tool_calls = []
            for call_data in tool_calls_data:
                if (
                    isinstance(call_data, dict)
                    and "name" in call_data
                    and "arguments" in call_data
                ):
                    tool_name = call_data["name"]
                    if tool_name in self.tool_map:
                        tool_call = ToolCall(
                            name=tool_name, arguments=call_data["arguments"]
                        )
                        tool_calls.append(tool_call)
                        logger.info(f"Generated tool call: {tool_name}")

            return tool_calls

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse generated tool calls: {e}")
            return []

    def generate_reply(
        self, thought: str, conversation_history: List[ChatMessage]
    ) -> str:
        """
        基于思考过程生成最终回复

        参数:
            thought: 思考过程字符串
            conversation_history: 对话历史消息列表

        返回:
            最终回复字符串
        """
        logger.info("Generating final reply based on thought process")

        # 使用模板渲染提示
        prompt = render_assistant_reply_prompt(thought, conversation_history)

        reply = chat_completion(
            prompt=prompt,
            system_message="",
            json_mode=False,
        )

        if not reply or reply.startswith("调用错误"):
            logger.error(f"Failed to generate reply: {reply}")
            return "我很乐意为您提供帮助！有什么可以效劳的吗？"

        logger.info(f"Generated reply: {reply[:100]}...")
        return reply.strip()


# ==================== 测试代码 ====================

if __name__ == "__main__":
    logger.info("🤖 Assistant Agent 测试")
    logger.info("=" * 50)

    from sloop.models import ChatMessage, ToolDefinition

    # 创建模拟工具
    mock_tools = [
        ToolDefinition(
            name="search_restaurants",
            description="Search for restaurants in a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "cuisine": {"type": "string", "description": "Type of cuisine"},
                },
                "required": ["city"],
            },
        ),
        ToolDefinition(
            name="book_restaurant",
            description="Book a table at a restaurant",
            parameters={
                "type": "object",
                "properties": {
                    "restaurant_id": {"type": "string", "description": "Restaurant ID"},
                    "date": {"type": "string", "description": "Booking date"},
                    "time": {"type": "string", "description": "Booking time"},
                    "party_size": {
                        "type": "integer",
                        "description": "Number of people",
                    },
                },
                "required": ["restaurant_id", "date", "time"],
            },
        ),
    ]

    # 创建模拟对话历史
    mock_history = [
        ChatMessage(role="user", content="我想在上海找一家意大利餐厅吃饭"),
        ChatMessage(
            role="assistant",
            content="我来帮你找上海的意大利餐厅。你想要什么样的价位或地点吗？",
        ),
        ChatMessage(role="user", content="市中心就可以，适合4个人"),
    ]

    logger.info("📋 测试数据:")
    logger.info(f"  可用工具数: {len(mock_tools)}")
    for tool in mock_tools:
        logger.info(f"    - {tool.name}: {tool.description}")
    logger.info(f"  对话历史: {len(mock_history)} 条消息")
    logger.info("")

    # 初始化助手智能体
    logger.info("🔧 初始化AssistantAgent...")
    assistant_agent = AssistantAgent(mock_tools)

    logger.info("🔧 测试工具调用解析...")
    try:
        # 测试解析功能
        mock_response = '我来帮你搜索上海的意大利餐厅。{"tool_name": "search_restaurants", "arguments": {"city": "上海", "cuisine": "意大利菜"}}'
        logger.info(f"📝 测试响应: {mock_response}")

        # 解析工具调用
        tool_calls = assistant_agent.parse_tool_calls(mock_response)
        if tool_calls:
            logger.info(f"🔧 检测到 {len(tool_calls)} 个工具调用:")
            for i, tool_call in enumerate(tool_calls, 1):
                logger.info(f"  {i}. {tool_call.name}: {tool_call.arguments}")
        else:
            logger.info("💬 纯文本响应，无工具调用")

    except Exception as e:
        logger.error(f"❌ 解析失败: {e}")

    logger.info("\n✅ Assistant Agent 测试完成！")
