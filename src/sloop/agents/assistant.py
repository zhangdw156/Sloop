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
