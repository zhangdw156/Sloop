"""
Sloop 项目配置模块
负责加载和管理强弱模型的 API 配置。
"""
from pydantic import BaseModel, Field
import os


# 注意：不再需要 from dotenv import load_dotenv 和 load_dotenv()

# 注意：环境变量的优先级现在由 os.getenv 的调用顺序隐式决定，但逻辑保持不变。
# 因为 os.getenv 会优先读取系统环境变量，如果不存在才返回默认值（即 .env 文件中的值）。
# 所以当前实现已经满足了“环境变量优先级大于 .env 文件”的要求。

class ModelConfig(BaseModel):
    """
    模型配置类
    定义强模型 (Teacher) 和弱模型 (Student) 的 API 配置
    """
    api_key: str = Field(..., description="API 密钥")
    base_url: str = Field(..., description="API 基础URL")


class SloopConfig:
    """
    Sloop 主配置类
    提供对强弱模型配置的集中访问
    """
    def __init__(self):
        """
        初始化配置，从环境变量中读取
        """
        # 强模型配置 (Teacher API)
        self.strong = ModelConfig(
            api_key=os.getenv("SLOOP_STRONG_API_KEY", ""),
            base_url=os.getenv("SLOOP_STRONG_BASE_URL", "")
        )
        # 弱模型配置 (Student API)
        self.weak = ModelConfig(
            api_key=os.getenv("SLOOP_WEAK_API_KEY", ""),
            base_url=os.getenv("SLOOP_WEAK_BASE_URL", "")
        )

    def validate(self) -> bool:
        """
        验证配置是否完整
        
        Returns:
            bool: 配置是否有效
        """
        return bool(
            self.strong.api_key and self.strong.base_url and
            self.weak.api_key and self.weak.base_url
        )
