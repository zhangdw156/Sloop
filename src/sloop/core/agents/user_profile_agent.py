"""
提供一个具体的用户画像生成器实现。
"""

from typing import Any, Dict

from openai import OpenAI

from sloop.core.agents.agent import UserProfileAgent


class SimpleUserProfileAgent(UserProfileAgent):
    """
    一个简单的用户画像生成器实现。
    """

    def __init__(self, client: OpenAI):
        """
        初始化。

        Args:
            client (OpenAI): OpenAI 客户端。
        """
        self.client = client

    def generate_profile(self) -> Dict[str, Any]:
        """
        生成用户画像。

        Args:
            problem (str): 需要解决的问题。

        Returns:
            Dict[str, Any]: 生成的用户画像。
        """
        # TODO: 实现用户画像生成逻辑
        # 这里需要根据问题生成一个合理的用户画像
        # 简化实现：返回一个固定的画像
        return {
            "age": 30,
            "occupation": "工程师",
            "interests": ["技术", "编程"],
            "personality": "理性",
        }
