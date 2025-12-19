"""
定义各种 Agent 的抽象基类。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class UserAgent(ABC):
    """
    用户代理，负责生成用户请求。
    """

    @abstractmethod
    def generate_request(self, problem: str, user_profile: Dict[str, Any]) -> str:
        """
        生成用户请求。

        Args:
            problem (str): 需要解决的问题。
            user_profile (Dict[str, Any]): 用户画像。

        Returns:
            str: 生成的用户请求。
        """
        pass


class AssistantAgent(ABC):
    """
    助手代理，负责生成助手的回复。
    """

    @abstractmethod
    def respond(self, user_message: str, conversation_history: str) -> str:
        """
        生成助手的回复。

        Args:
            user_message (str): 用户的消息。
            conversation_history (str): 对话历史。

        Returns:
            str: 助手的回复。
        """
        pass


class ServiceAgent(ABC):
    """
    服务代理，负责执行服务调用。
    """

    @abstractmethod
    def execute_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行服务调用。

        Args:
            tool_call (Dict[str, Any]): 服务调用的请求。

        Returns:
            Dict[str, Any]: 服务调用的执行结果。
        """
        pass


class Planner(ABC):
    """
    规划器，负责规划整个对话流程。
    """

    @abstractmethod
    def plan_dialogue(self, problem: str, apis: list) -> Dict[str, Any]:
        """
        规划整个对话流程。

        Args:
            problem (str): 需要解决的问题。
            apis (list): 可用的 API 列表。

        Returns:
            Dict[str, Any]: 包含对话流程规划的字典。
        """
        pass


class UserProfileAgent(ABC):
    """
    用户画像生成器，负责生成用户画像。
    """

    @abstractmethod
    def generate_profile(self, problem: str) -> Dict[str, Any]:
        """
        生成用户画像。

        Args:
            problem (str): 需要解决的问题。

        Returns:
            Dict[str, Any]: 生成的用户画像。
        """
        pass
