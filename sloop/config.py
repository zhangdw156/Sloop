"""
配置管理模块

负责加载环境变量和配置参数，支持通过.env文件进行配置。
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

# 加载.env文件
load_dotenv()

# 延迟导入logger，避免循环导入
try:
    from sloop.utils.logger import logger
except ImportError:
    logger = None


# 延迟导入logger，避免循环导入
def _get_logger():
    if logger is not None:
        return logger
    return logging.getLogger(__name__)


class Settings(BaseModel):
    """应用配置类"""

    # LLM配置
    llm_provider: str = Field(default="openai", env="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4o-mini", validation_alias="MODEL_NAME")
    openai_api_key: Optional[str] = Field(default=None, env="API_KEY")
    openai_api_base: Optional[str] = Field(default=None, env="API_BASE")
    temperature: float = Field(default=0.7, env="TEMPERATURE")

    # 系统配置
    max_tokens: int = Field(default=2048, env="MAX_TOKENS")
    timeout: int = Field(default=60, env="TIMEOUT")

    # Embedding 配置
    embedding_provider: str = Field(default="openai", env="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    embedding_api_key: Optional[str] = Field(default=None, env="EMBEDDING_API_KEY")
    embedding_base_url: Optional[str] = Field(default=None, env="EMBEDDING_API_BASE")

    @model_validator(mode="after")
    def set_defaults(self):
        """设置默认值"""
        if not self.embedding_provider:
            self.embedding_provider = self.llm_provider
        if not self.embedding_api_key:
            self.embedding_api_key = self.openai_api_key
        if not self.embedding_base_url:
            self.embedding_base_url = self.openai_api_base
        return self

    def get_llm_model_id(self) -> str:
        """获取 LLM 模型 ID，用于 litellm 调用"""
        if self.llm_provider == "openai":
            return self.llm_model
        return f"{self.llm_provider}/{self.llm_model}"

    def validate(self) -> bool:
        """验证配置是否有效"""
        if not self.openai_api_key:
            _get_logger().error("❌ 错误: 未配置 API_KEY")
            return False

        if self.temperature < 0.0 or self.temperature > 2.0:
            _get_logger().error("❌ 错误: TEMPERATURE 必须在 0.0-2.0 之间")
            return False

        return True

    def get_safe_display(self) -> dict:
        """获取安全的配置显示（隐藏敏感信息）"""
        return {
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "openai_api_key": f"{self.openai_api_key[:4]}***"
            if self.openai_api_key
            else "未设置",
            "openai_api_base": self.openai_api_base or "默认",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_api_key": f"{self.embedding_api_key[:4]}***" if self.embedding_api_key else "未设置",
            "embedding_base_url": self.embedding_base_url or "默认",
        }


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取全局配置实例"""
    return settings


def reload_settings() -> Settings:
    """重新加载配置（用于测试或动态更新）"""
    global settings
    settings = Settings()
    return settings


if __name__ == "__main__":
    logger = _get_logger()
    logger.info("🔧 配置验证")
    logger.info("=" * 50)

    # 验证配置
    if settings.validate():
        logger.info("✅ 配置验证通过")
        logger.info("\n📋 当前配置:")
        safe_config = settings.get_safe_display()
        for key, value in safe_config.items():
            logger.info(f"  {key}: {value}")
    else:
        logger.error("❌ 配置验证失败")
        logger.info("\n请检查以下环境变量:")
        logger.info("  - API_KEY: 必需")
        logger.info("  - LLM_PROVIDER: 可选，默认 openai")
        logger.info("  - MODEL_NAME (或 LLM_MODEL): 可选，默认 gpt-4o-mini")
        logger.info("  - API_BASE: 可选")
        logger.info("  - TEMPERATURE: 可选，默认 0.7")
        logger.info("  - MAX_TOKENS: 可选，默认 4096")
        logger.info("  - TIMEOUT: 可选，默认 60")
        logger.info("  - EMBEDDING_PROVIDER: 可选，默认 openai")
        logger.info("  - EMBEDDING_MODEL: 可选，默认 text-embedding-3-small")
        logger.info("  - EMBEDDING_API_KEY: 可选，默认复用 OPENAI_API_KEY")
        logger.info("  - EMBEDDING_BASE_URL: 可选")
