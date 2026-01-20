"""
环境变量配置管理模块
负责读取和管理项目根目录下的.env文件
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


class EnvConfig:
    """环境变量配置管理类"""

    def __init__(self, env_file: Optional[str] = None):
        """
        初始化环境配置

        Args:
            env_file: .env文件路径，如果为None则使用项目根目录的.env文件
        """
        # 确定.env文件路径
        if env_file:
            self.env_file = Path(env_file)
        else:
            # 默认使用项目根目录的.env文件
            self.env_file = Path(__file__).parent.parent.parent / ".env"

        # 加载环境变量
        self.load_environment()

    def load_environment(self) -> None:
        """加载环境变量"""
        if self.env_file.exists():
            load_dotenv(dotenv_path=self.env_file, override=True)
        else:
            raise FileNotFoundError(f".env文件不存在: {self.env_file}")

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        获取环境变量值

        Args:
            key: 环境变量键名
            default: 默认值

        Returns:
            环境变量值，如果不存在则返回默认值
        """
        return os.getenv(key, default)

    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """
        获取整数类型的环境变量值

        Args:
            key: 环境变量键名
            default: 默认值

        Returns:
            整数类型的环境变量值
        """
        value = self.get(key)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                pass
        return default

    def get_bool(self, key: str, default: Optional[bool] = None) -> Optional[bool]:
        """
        获取布尔类型的环境变量值

        Args:
            key: 环境变量键名
            default: 默认值

        Returns:
            布尔类型的环境变量值
        """
        value = self.get(key)
        if value is not None:
            return value.lower() in ("true", "1", "yes", "on")
        return default


# 创建全局配置实例
env_config = EnvConfig()
