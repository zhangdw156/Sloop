"""
Sloop配置模块
支持强弱模型配置和CrewAI设置
"""

from typing import Optional
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()


class ModelConfig(BaseModel):
    """模型配置"""
    api_key: str = Field(..., description="API密钥")
    base_url: str = Field(..., description="API基础URL")
    model_name: str = Field(..., description="模型名称")

    @classmethod
    def from_env(cls, prefix: str) -> "ModelConfig":
        """从环境变量创建配置"""
        return cls(
            api_key=os.getenv(f"{prefix}_API_KEY", ""),
            base_url=os.getenv(f"{prefix}_BASE_URL", ""),
            model_name=os.getenv(f"{prefix}_MODEL_NAME", "gpt-4o")
        )


class SloopConfig(BaseModel):
    """Sloop完整配置"""
    strong: ModelConfig = Field(default_factory=lambda: ModelConfig.from_env("SLOOP_STRONG"))
    weak: Optional[ModelConfig] = Field(default_factory=lambda: ModelConfig.from_env("SLOOP_WEAK"))

    # 系统配置
    verbose: bool = Field(default_factory=lambda: os.getenv("SLOOP_VERBOSE", "true").lower() == "true",
                         description="是否启用详细输出")

    def validate(self) -> bool:
        """验证配置是否有效"""
        if not self.strong.api_key or not self.strong.base_url:
            return False
        if not self.strong.model_name:
            self.strong.model_name = "gpt-4o"  # 默认值
        return True

    def get_model_config(self, model_type: str = "strong") -> ModelConfig:
        """获取指定类型的模型配置"""
        if model_type == "strong":
            return self.strong
        elif model_type == "weak" and self.weak:
            return self.weak
        else:
            return self.strong  # 默认使用强模型


# 全局配置实例
config = SloopConfig()
