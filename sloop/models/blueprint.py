"""
蓝图数据模型：任务蓝图定义

包含用户意图、必需工具、执行路径和状态定义。
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from sloop.models.schema import UserPersona


class Blueprint(BaseModel):
    """
    任务蓝图：定义对话生成的任务大纲

    在正式对话前先生成蓝图，确保多轮对话有明确的终点和逻辑一致性。
    """

    intent: str = Field(..., description="用户的高层意图描述")
    required_tools: List[str] = Field(..., description="必须使用的工具名称列表")
    ground_truth: List[str] = Field(
        ..., description="预期的正确调用序列（工具名称列表）"
    )
    initial_state: Dict[str, Any] = Field(..., description="环境初始状态字典")
    expected_state: Dict[str, Any] = Field(..., description="结束时期望状态字典")
    persona: Optional[UserPersona] = Field(
        None, description="用户画像，用于生成多样化的用户行为"
    )

    class Config:
        extra = "allow"  # 允许额外字段以支持扩展

    def validate_state_transition(self) -> bool:
        """
        验证初始状态和期望状态的键是否一致

        返回: bool - 状态转换是否有效
        """
        return set(self.initial_state.keys()) == set(self.expected_state.keys())

    def get_tool_sequence_str(self) -> str:
        """
        获取工具调用序列的字符串表示

        返回: str - 工具序列描述
        """
        return " -> ".join(self.ground_truth)

    def __str__(self) -> str:
        """蓝图的字符串表示"""
        return f"Blueprint(intent='{self.intent}', tools={self.required_tools}, sequence={self.get_tool_sequence_str()})"
