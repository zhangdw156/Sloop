"""
核心数据模型：符合 OpenAI 规范的工具定义和消息格式

使用 Pydantic v2 进行数据验证和类型提示。
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ToolParameterProperty(BaseModel):
    """工具参数的属性定义"""
    type: str = Field(..., description="参数类型，如 'string', 'number', 'boolean'")
    description: Optional[str] = Field(None, description="参数描述")
    enum: Optional[List[str]] = Field(None, description="枚举值列表")
    items: Optional[Dict[str, Any]] = Field(None, description="数组项的定义（用于数组类型）")


class ToolParameters(BaseModel):
    """工具参数的整体定义"""
    type: str = Field("object", description="参数结构类型，通常为 'object'")
    properties: Dict[str, ToolParameterProperty] = Field(..., description="参数属性字典")
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
    role: str = Field(..., description="消息角色：'user', 'assistant', 'tool', 'system'")
    content: Optional[str] = Field(None, description="消息内容")
    tool_call: Optional[ToolCall] = Field(None, description="工具调用（仅assistant消息）")
    tool_call_id: Optional[str] = Field(None, description="工具调用ID（仅tool消息）")

    class Config:
        extra = "allow"  # 允许额外字段以支持扩展
