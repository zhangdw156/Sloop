"""
配置管理模块

负责加载环境变量和配置参数，支持通过.env文件进行配置。
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

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


@dataclass
class Settings:
    """应用配置类"""

    # LLM配置
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None
    temperature: float = 0.7

    # 系统配置
    max_tokens: int = 2048
    timeout: int = 60

    # Embedding 配置
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: Optional[str] = None
    embedding_base_url: Optional[str] = None

    def __post_init__(self):
        """从环境变量初始化"""
        # LLM配置
        self.llm_provider = os.getenv("LLM_PROVIDER", self.llm_provider)
        self.llm_model = os.getenv("MODEL_NAME", self.llm_model)
        self.openai_api_key = os.getenv("API_KEY", self.openai_api_key)
        self.openai_api_base = os.getenv("API_BASE", self.openai_api_base)

        # 温度参数
        try:
            temp_str = os.getenv("TEMPERATURE")
            if temp_str:
                self.temperature = float(temp_str)
        except (ValueError, TypeError):
            pass  # 使用默认值

        # 系统配置
        try:
            max_tokens_str = os.getenv("MAX_TOKENS")
            if max_tokens_str:
                self.max_tokens = int(max_tokens_str)
        except (ValueError, TypeError):
            pass

        try:
            timeout_str = os.getenv("TIMEOUT")
            if timeout_str:
                self.timeout = int(timeout_str)
        except (ValueError, TypeError):
            pass

        # Embedding配置
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", self.embedding_provider)
        self.embedding_model = os.getenv("EMBEDDING_MODEL", self.embedding_model)
        self.embedding_api_key = os.getenv("EMBEDDING_API_KEY", self.embedding_api_key)
        self.embedding_base_url = os.getenv("EMBEDDING_API_BASE", self.embedding_base_url)

        # 如果embedding参数未设置，则复用llm参数
        if not self.embedding_provider:
            self.embedding_provider = self.llm_provider
        if not self.embedding_api_key:
            self.embedding_api_key = self.openai_api_key
        if not self.embedding_base_url:
            self.embedding_base_url = self.openai_api_base

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
        logger.info("  - EMBEDDING_PROVIDER: 可选，默认复用 LLM_PROVIDER")
        logger.info("  - EMBEDDING_MODEL: 可选，默认 text-embedding-3-small")
        logger.info("  - EMBEDDING_API_KEY: 可选，默认复用 API_KEY")
        logger.info("  - EMBEDDING_API_BASE: 可选，默认复用 API_BASE")
