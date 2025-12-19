"""
提供一个具体的助手代理实现。
"""
from openai import OpenAI
from ..agent import AssistantAgent


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

    def respond(
        self, 
        user_message: str, 
        conversation_history: str
    ) -> str:
        """
        根据用户消息和对话历史生成助手的回复。
        
        Args:
            user_message (str): 用户的消息。
            conversation_history (str): 对话历史。
            
        Returns:
            str: 助手的回复。
        """
        # TODO: 实现助手回复生成逻辑
        prompt = (
            f"你是一个助手，请根据以下对话历史和用户的新消息进行回复。"
            f"对话历史: {conversation_history}\n用户: {user_message}"
        )
        prompt = (
            f"你是一个助手，请根据以下对话历史和用户的新消息进行回复。"
            f"对话历史: {conversation_history}\n用户: {user_message}"
        )
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512
        )
        return response.choices[0].message.content
