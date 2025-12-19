"""
提供一个具体的用户代理实现。
"""

from typing import Any, Dict

from openai import OpenAI

from sloop.core.agents.agent import UserAgent, UserProfileAgent
from sloop.core.prompts.user_agent import GENERATE_REQUEST_PROMPT, GENERATE_USER_MESSAGE_PROMPT


class SimpleUserAgent(UserAgent):
    """
    一个简单的用户代理实现。
    """

    def __init__(self, client: OpenAI, user_profile_agent: UserProfileAgent):
        """
        初始化。

        Args:
            client (OpenAI): OpenAI 客户端。
            user_profile_agent (UserProfileAgent): 用户画像生成器。
        """
        self.client = client
        self.user_profile_agent = user_profile_agent

    def generate_request(self, problem: str, user_profile: Dict[str, Any]) -> str:
        """
        生成用户请求。

        Args:
            problem (str): 需要解决的问题。
            user_profile (Dict[str, Any]): 用户画像。

        Returns:
            str: 生成的用户请求。
        """
        # TODO: 实现用户请求生成逻辑
        # 确保将问题包装在 Prompt 中请求 Mock API，而不是直接透传
        prompt = GENERATE_REQUEST_PROMPT.format(problem=problem, user_profile=user_profile)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=256,
            timeout=10.0,
        )
        # 返回模型生成的内容，而不是直接返回问题
        return response.choices[0].message.content

    def generate_user_message(self, problem: str, context: Dict[str, Any]) -> str:
        """
        根据问题和上下文生成用户消息。

        Args:
            problem (str): 需要解决的问题。
            context (Dict[str, Any]): 对话上下文。

        Returns:
            str: 生成的用户消息。
        """
        # TODO: 实现用户消息生成逻辑
        prompt = GENERATE_USER_MESSAGE_PROMPT.format(problem=problem, context=context)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=256,
        )
        return response.choices[0].message.content
