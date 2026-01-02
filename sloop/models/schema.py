"""
核心数据模型：符合 OpenAI 规范的工具定义和消息格式

使用 Pydantic v2 进行数据验证和类型提示。
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolParameterProperty(BaseModel):
    """工具参数的属性定义"""

    type: str = Field(..., description="参数类型，如 'string', 'number', 'boolean'")
    description: Optional[str] = Field(None, description="参数描述")
    enum: Optional[List[str]] = Field(None, description="枚举值列表")
    items: Optional[Dict[str, Any]] = Field(
        None, description="数组项的定义（用于数组类型）"
    )


class ToolParameters(BaseModel):
    """工具参数的整体定义"""

    type: str = Field("object", description="参数结构类型，通常为 'object'")
    properties: Dict[str, ToolParameterProperty] = Field(
        ..., description="参数属性字典"
    )
    required: Optional[List[str]] = Field(None, description="必需的参数列表")


class ToolDefinition(BaseModel):
    """OpenAI 规范的工具定义"""

    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="工具描述")
    parameters: ToolParameters = Field(..., description="工具参数定义")

    class Config:
        extra = "allow"  # 允许额外字段


class ToolCall(BaseModel):
    """工具调用定义"""

    name: str = Field(..., description="调用的工具名称")
    arguments: Dict[str, Any] = Field(..., description="工具调用参数")


class ChatMessage(BaseModel):
    """聊天消息定义"""

    role: str = Field(
        ..., description="消息角色：'user', 'assistant', 'tool', 'system'"
    )
    content: Optional[str] = Field(None, description="消息内容")
    tool_call: Optional[ToolCall] = Field(
        None, description="单个工具调用（仅assistant消息，向后兼容）"
    )
    tool_calls: Optional[List[ToolCall]] = Field(
        None, description="并行工具调用列表（仅assistant消息）"
    )
    tool_call_id: Optional[str] = Field(None, description="工具调用ID（仅tool消息）")

    class Config:
        extra = "allow"  # 允许额外字段以支持扩展


class UserPersona(BaseModel):
    """用户画像定义"""

    name: str = Field(..., description="画像名称")
    description: str = Field(..., description="画像描述")
    behavior_traits: List[str] = Field(..., description="行为特征列表")
    communication_style: str = Field(..., description="沟通风格描述")
    expertise_level: str = Field(
        "intermediate",
        description="专业程度：'novice', 'intermediate', 'expert'"
    )
    patience_level: str = Field(
        "normal",
        description="耐心程度：'impatient', 'normal', 'patient'"
    )
    complexity_threshold: int = Field(
        2,
        description="能处理的工具链复杂度阈值，超过此值会显得困惑"
    )
