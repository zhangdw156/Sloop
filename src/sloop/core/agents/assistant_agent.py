"""
提供一个具体的助手代理实现。
"""

from openai import OpenAI

from sloop.core.agents.agent import AssistantAgent
from sloop.core.prompts.assistant_agent import GENERATE_RESPONSE_PROMPT_FOR_TOOL_CALL, GENERATE_RESPONSE_PROMPT_FOR_SUMMARY


class SimpleAssistantAgent(AssistantAgent):
    """
    一个简单的助手代理实现。
    """

    def __init__(self, client: OpenAI):
        """
        初始化。

        Args:
            client (OpenAI): OpenAI 客户端。
        """
        self.client = client

    def respond(self, user_message: str, conversation_history: list) -> str:
        """
        根据用户消息和对话历史生成助手的回复。
        此方法会根据对话历史的上下文决定使用哪个提示词。

        Args:
            user_message (str): 用户的消息。
            conversation_history (list): 结构化的对话历史列表。

        Returns:
            str: 助手的回复。
        """
        # 检查对话历史中是否已存在 'tool' 角色的消息
        has_tool_call = any(msg.get("role") == "tool" for msg in conversation_history)

        if has_tool_call:
            # 如果存在 tool 消息，使用总结提示词，禁止输出 XML 标签
            prompt = GENERATE_RESPONSE_PROMPT_FOR_SUMMARY.format(conversation_history=conversation_history)
            # 为了确保 GCP 的确定性，探测时应设置 temperature=0，但此处为通用逻辑
            temperature = 0.7
        else:
            # 如果没有 tool 消息，使用工具调用提示词
            prompt = GENERATE_RESPONSE_PROMPT_FOR_TOOL_CALL.format(conversation_history=conversation_history, user_message=user_message)
            temperature = 0.7

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=512,
            timeout=10.0,
        )
        return response.choices[0].message.content
