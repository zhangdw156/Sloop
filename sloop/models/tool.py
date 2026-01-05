# sloop/models/tool.py
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

class ToolParameters(BaseModel):
    """定义工具的参数结构 (JSON Schema)"""
    type: str = "object"
    properties: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)

class ToolDefinition(BaseModel):
    """定义一个完整的工具"""
    name: str
    description: str
    parameters: ToolParameters
    category: str = "general"

    class Config:
        extra = "allow"  # 允许额外的字段（如 embedding 缓存等）