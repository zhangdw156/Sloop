"""
提供一个具体的助手代理实现。
"""

from openai import OpenAI

from sloop.core.agents.agent import AssistantAgent
from sloop.core.prompts.assistant_agent import GENERATE_RESPONSE_PROMPT


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

    def respond(self, user_message: str, conversation_history: str) -> str:
        """
        根据用户消息和对话历史生成助手的回复。

        Args:
            user_message (str): 用户的消息。
            conversation_history (str): 对话历史。

        Returns:
            str: 助手的回复。
        """
        # TODO: 实现助手回复生成逻辑
        prompt = GENERATE_RESPONSE_PROMPT.format(conversation_history=conversation_history, user_message=user_message)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512,
            timeout=10.0,
        )
        return response.choices[0].message.content
