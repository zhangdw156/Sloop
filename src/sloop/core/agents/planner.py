"""
提供一个具体的规划器实现。
"""

from typing import Any, Dict

from openai import OpenAI

from sloop.core.agents.agent import Planner


class SimplePlanner(Planner):
    """
    一个简单的规划器实现。
    """

    def __init__(self, client: OpenAI):
        """
        初始化。

        Args:
            client (OpenAI): OpenAI 客户端。
        """
        self.client = client

    def plan_dialogue(self, problem: str, apis: list) -> Dict[str, Any]:
        """
        规划整个对话流程。

        Args:
            problem (str): 需要解决的问题。
            apis (list): 可用的 API 列表。

        Returns:
            Dict[str, Any]: 包含对话流程规划的字典。
        """
        # TODO: 实现对话流程规划逻辑
        # 这里需要规划多轮对话，决定何时调用哪个 API
        # 简化实现：返回一个固定的规划
        return {
            "problem": problem,
            "steps": [
                {
                    "action": "call_api",
                    "api": apis[0].name if apis else "unknown_api",
                }
            ],
        }
