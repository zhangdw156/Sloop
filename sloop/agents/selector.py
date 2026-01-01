"""
裁判智能体 (Selector Agent)

负责在候选工具中选择最佳的下一个工具，或决定结束任务。
基于当前的工具链条和候选工具列表，使用 LLM 进行智能决策。
"""

from typing import List, Optional

from sloop.config import get_settings
from sloop.models import ToolDefinition
from sloop.utils.llm import chat_completion
from sloop.utils.logger import logger
from sloop.utils.template import render_selector_prompt


class SelectorAgent:
    """
    裁判智能体

    分析当前工具链条，选择最合适的下一个工具或决定结束任务。
    """

    def __init__(self):
        """
        初始化裁判智能体
        """
        self.settings = get_settings()

    def select_next_tool(
        self,
        current_chain: List[str],
        candidates: List[ToolDefinition]
    ) -> Optional[str]:
        """
        选择下一个工具

        参数:
            current_chain: 当前已执行的工具名称列表
            candidates: 候选工具定义列表

        返回:
            选中的工具名称，或 None 表示结束任务
        """
        if not candidates:
            return None

        # 使用模板渲染提示
        user_prompt = render_selector_prompt(current_chain, candidates)

        # 调用 LLM（系统提示已在模板中）
        response = chat_completion(
            prompt=user_prompt,
            system_message="",
            json_mode=False,
        )

        if not response or response.startswith("调用错误"):
            # LLM 调用失败，默认结束任务
            return None

        # 清理响应
        result = response.strip()

        # 检查是否选择结束
        if result.upper() == "FINISH":
            return None

        # 检查是否是有效的工具名称
        valid_names = {tool.name for tool in candidates}
        if result in valid_names:
            return result

        # 如果不是有效名称，尝试提取工具名称
        for tool in candidates:
            if tool.name in result:
                return tool.name

        # 如果无法识别，默认结束任务
        return None
