"""
状态数据模型：环境状态和对话上下文

用于有状态服务模拟和FSM状态传递。
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from .schema import ChatMessage, ToolCall


class EnvState(BaseModel):
    """
    虚拟环境状态：维护有状态的工具调用环境

    模拟真实API的状态变更，避免LLM产生逻辑幻觉。
    """
    state: Dict[str, Any] = Field(default_factory=dict, description="当前环境状态字典")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="状态变更历史记录")

    def get(self, key: str, default: Any = None) -> Any:
        """获取状态值"""
        return self.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """设置状态值并记录历史"""
        old_value = self.state.get(key)
        self.state[key] = value

        # 记录变更历史
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "key": key,
            "old_value": old_value,
            "new_value": value,
            "action": "set"
        })

    def update(self, updates: Dict[str, Any]) -> None:
        """批量更新状态"""
        for key, value in updates.items():
            self.set(key, value)

    def reset(self) -> None:
        """重置状态"""
        self.state.clear()
        self.history.clear()

    def validate_transition(self, expected_state: Dict[str, Any]) -> bool:
        """
        验证当前状态是否符合期望状态

        参数:
            expected_state: 期望的状态字典

        返回: bool - 是否匹配
        """
        return self.state == expected_state

    def __str__(self) -> str:
        """状态的字符串表示"""
        return f"EnvState({self.state})"


class ConversationContext(BaseModel):
    """
    对话上下文：用于FSM传递上下文信息

    包含对话历史、轮次计数、当前状态等。
    """
    conversation_id: str = Field(..., description="对话唯一标识")
    blueprint_id: Optional[str] = Field(None, description="关联的蓝图ID")

    messages: List[ChatMessage] = Field(default_factory=list, description="对话消息历史")
    turn_count: int = Field(default=0, description="当前对话轮次")

    env_state: EnvState = Field(default_factory=EnvState, description="当前环境状态")
    initial_state: Dict[str, Any] = Field(default_factory=dict, description="初始环境状态快照")

    current_user_intent: Optional[str] = Field(None, description="当前用户意图")
    pending_tool_calls: List[ToolCall] = Field(default_factory=list, description="待处理的工具调用")

    max_turns: int = Field(default=10, description="最大对话轮次限制")
    is_completed: bool = Field(default=False, description="对话是否完成")

    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="最后更新时间")

    def add_message(self, message: ChatMessage) -> None:
        """添加消息到对话历史"""
        self.messages.append(message)
        self.updated_at = datetime.now()

    def increment_turn(self) -> None:
        """增加轮次计数"""
        self.turn_count += 1
        self.updated_at = datetime.now()

    def can_continue(self) -> bool:
        """检查是否可以继续对话"""
        return not self.is_completed and self.turn_count < self.max_turns

    def complete_conversation(self) -> None:
        """标记对话完成"""
        self.is_completed = True
        self.updated_at = datetime.now()

    def get_last_message(self) -> Optional[ChatMessage]:
        """获取最后一条消息"""
        return self.messages[-1] if self.messages else None

    def get_tool_call_history(self) -> List[ToolCall]:
        """获取工具调用历史"""
        tool_calls = []
        for msg in self.messages:
            if msg.tool_call:
                tool_calls.append(msg.tool_call)
        return tool_calls

    def __str__(self) -> str:
        """上下文的字符串表示"""
        return f"ConversationContext(id='{self.conversation_id}', turns={self.turn_count}, completed={self.is_completed})"
