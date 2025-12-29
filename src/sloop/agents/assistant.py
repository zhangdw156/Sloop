"""
助手模拟器 (Assistant Agent)

模拟被测试的助手模型，根据工具定义和对话历史决定下一步行动。
"""

import logging
import json
import re
from typing import List, Optional, Dict, Any
from ..models import ToolDefinition, ChatMessage, ToolCall
from ..utils.llm import chat_completion
from ..utils.template import render_assistant_prompt

logger = logging.getLogger(__name__)


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

    def generate_response(
        self,
        conversation_history: List[ChatMessage]
    ) -> str:
        """
        生成助手响应

        参数:
            conversation_history: 对话历史消息列表

        返回:
            助手响应字符串，可能包含工具调用信息
        """
        logger.info("Generating assistant response")

        # 构造提示
        prompt = render_assistant_prompt(self.tools, conversation_history)

        # 调用LLM生成响应
        response = chat_completion(
            prompt=prompt,
            system_message="You are a helpful AI assistant with access to various tools. Use tools when appropriate to help the user.",
            json_mode=False  # 让模型自由输出，可能包含工具调用
        )

        if not response or response.startswith("调用错误"):
            logger.error(f"Failed to generate assistant response: {response}")
            return "I'm sorry, I encountered an error. How can I help you?"  # 默认响应

        logger.info(f"Generated assistant response: {response[:100]}...")
        return response.strip()

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
                    tool_call = ToolCall(
                        name=tool_name,
                        arguments=arguments
                    )
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
                        tool_call = ToolCall(
                            name=tool_name,
                            arguments=arguments
                        )
                        tool_calls.append(tool_call)
                        logger.info(f"Parsed function call: {tool_name}")
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse function call arguments: {args_str}")

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

    def generate_thought(self, conversation_history: List[ChatMessage]) -> str:
        """
        生成助手思考过程 (Chain of Thought)

        参数:
            conversation_history: 对话历史消息列表

        返回:
            思考过程字符串
        """
        logger.info("Generating assistant thought process (CoT)")

        # 构造思考提示
        prompt = f"""Based on the conversation history, generate a step-by-step reasoning process for how to respond to the user's latest message.

Conversation History:
{self._format_history(conversation_history)}

Please provide a detailed thought process considering:
1. What the user is asking for
2. What information you have
3. What tools might be needed
4. How to structure your response

Thought Process:"""

        # 调用LLM生成思考过程
        thought = chat_completion(
            prompt=prompt,
            system_message="You are a reasoning AI that generates detailed thought processes. Be thorough and logical.",
            json_mode=False
        )

        if not thought or thought.startswith("调用错误"):
            logger.error(f"Failed to generate thought: {thought}")
            return "I need to analyze the user's request and determine the best way to respond."

        logger.info(f"Generated thought: {thought[:100]}...")
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

        prompt = f"""Based on the following thought process, determine if tools are needed to fulfill the user's request.

Thought Process:
{thought}

Available Tools:
{self._format_tools()}

Answer with only 'YES' or 'NO': Do tools need to be called?"""

        decision = chat_completion(
            prompt=prompt,
            system_message="You are a decision-making AI. Answer only with YES or NO.",
            json_mode=False
        ).strip().upper()

        needs_tools = decision.startswith('YES')
        logger.info(f"Tool use decision: {needs_tools}")
        return needs_tools

    def generate_tool_calls(self, thought: str, tools: List[ToolDefinition]) -> List[ToolCall]:
        """
        基于思考过程生成工具调用

        参数:
            thought: 思考过程字符串
            tools: 可用的工具列表

        返回:
            工具调用列表
        """
        logger.info("Generating tool calls based on thought process")

        prompt = f"""Based on the thought process, generate the appropriate tool calls in JSON format.

Thought Process:
{thought}

Available Tools:
{self._format_tools()}

Generate tool calls as a JSON array. Each tool call should have 'name' and 'arguments' fields.
If no tools are needed, return an empty array.

Tool Calls:"""

        response = chat_completion(
            prompt=prompt,
            system_message="You are a tool-calling AI. Generate tool calls in valid JSON format.",
            json_mode=True
        )

        try:
            tool_calls_data = json.loads(response)
            if not isinstance(tool_calls_data, list):
                tool_calls_data = [tool_calls_data]

            tool_calls = []
            for call_data in tool_calls_data:
                if isinstance(call_data, dict) and 'name' in call_data and 'arguments' in call_data:
                    tool_name = call_data['name']
                    if tool_name in self.tool_map:
                        tool_call = ToolCall(
                            name=tool_name,
                            arguments=call_data['arguments']
                        )
                        tool_calls.append(tool_call)
                        logger.info(f"Generated tool call: {tool_name}")

            return tool_calls

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse generated tool calls: {e}")
            return []

    def generate_reply(self, thought: str, conversation_history: List[ChatMessage]) -> str:
        """
        基于思考过程生成最终回复

        参数:
            thought: 思考过程字符串
            conversation_history: 对话历史消息列表

        返回:
            最终回复字符串
        """
        logger.info("Generating final reply based on thought process")

        prompt = f"""Based on your thought process, generate a helpful and natural response to the user.

Thought Process:
{thought}

Conversation History:
{self._format_history(conversation_history)}

Generate a response that:
1. Addresses the user's needs
2. Is helpful and friendly
3. Uses information from the thought process
4. Does not mention internal reasoning

Response:"""

        reply = chat_completion(
            prompt=prompt,
            system_message="You are a helpful AI assistant. Generate natural, helpful responses.",
            json_mode=False
        )

        if not reply or reply.startswith("调用错误"):
            logger.error(f"Failed to generate reply: {reply}")
            return "I'm here to help! How can I assist you?"

        logger.info(f"Generated reply: {reply[:100]}...")
        return reply.strip()

    def _format_history(self, history: List[ChatMessage]) -> str:
        """格式化对话历史"""
        formatted = []
        for msg in history:
            formatted.append(f"{msg.role}: {msg.content}")
        return "\n".join(formatted)

    def _format_tools(self) -> str:
        """格式化工具列表"""
        formatted = []
        for tool in self.tools:
            formatted.append(f"- {tool.name}: {tool.description}")
            if tool.parameters and 'properties' in tool.parameters:
                for prop_name, prop_info in tool.parameters['properties'].items():
                    formatted.append(f"  * {prop_name}: {prop_info.get('description', 'No description')}")
        return "\n".join(formatted)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("🤖 Assistant Agent 测试")
    print("=" * 50)

    from ..models import ToolDefinition, ChatMessage

    # 创建模拟工具
    mock_tools = [
        ToolDefinition(
            name="search_restaurants",
            description="Search for restaurants in a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "cuisine": {"type": "string", "description": "Type of cuisine"}
                },
                "required": ["city"]
            }
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
                    "party_size": {"type": "integer", "description": "Number of people"}
                },
                "required": ["restaurant_id", "date", "time"]
            }
        )
    ]

    # 创建模拟对话历史
    mock_history = [
        ChatMessage(role="user", content="我想在上海找一家意大利餐厅吃饭"),
        ChatMessage(role="assistant", content="我来帮你找上海的意大利餐厅。你想要什么样的价位或地点吗？"),
        ChatMessage(role="user", content="市中心就可以，适合4个人"),
    ]

    print("📋 测试数据:")
    print(f"  可用工具数: {len(mock_tools)}")
    for tool in mock_tools:
        print(f"    - {tool.name}: {tool.description}")
    print(f"  对话历史: {len(mock_history)} 条消息")
    print()

    # 初始化助手智能体
    print("🔧 初始化AssistantAgent...")
    assistant_agent = AssistantAgent(mock_tools)

    print("💭 生成助手响应...")
    try:
        response = assistant_agent.generate_response(mock_history)

        print("✅ 生成成功！")
        print(f"📝 响应内容: {response}")

        # 解析工具调用
        tool_calls = assistant_agent.parse_tool_calls(response)
        if tool_calls:
            print(f"🔧 检测到 {len(tool_calls)} 个工具调用:")
            for i, tool_call in enumerate(tool_calls, 1):
                print(f"  {i}. {tool_call.tool_name}: {tool_call.arguments}")
        else:
            print("💬 纯文本响应，无工具调用")

    except Exception as e:
        print(f"❌ 生成失败: {e}")

        # 如果LLM调用失败，提供模拟结果
        print("\n🔧 提供模拟助手响应:")
        mock_response = '我来帮你搜索上海的意大利餐厅。{"tool_name": "search_restaurants", "arguments": {"city": "上海", "cuisine": "意大利菜"}}'
        print(mock_response)

        # 测试解析
        tool_calls = assistant_agent.parse_tool_calls(mock_response)
        if tool_calls:
            print(f"🔧 解析出 {len(tool_calls)} 个工具调用:")
            for tool_call in tool_calls:
                print(f"  - {tool_call.name}: {tool_call.arguments}")

    print("\n✅ Assistant Agent 测试完成！")
