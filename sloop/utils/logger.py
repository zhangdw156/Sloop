import os

from loguru import logger
from tqdm import tqdm


def tqdm_sink(msg):
    """自定义 sink，使用 tqdm.write 输出日志，避免与进度条冲突"""
    tqdm.write(str(msg), end="")


def setup_logging():
    """
    设置日志配置

    Console: 只显示 WARNING/ERROR 级别日志，避免干扰进度条
    File: 记录所有 DEBUG/INFO 级别日志，包含线程信息
    """
    # 确保 logs 目录存在
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "logs",
    )
    os.makedirs(log_dir, exist_ok=True)

    # 移除默认的 loguru handler
    logger.remove()

    # 添加控制台输出（只显示 WARNING/ERROR）
    logger.add(
        tqdm_sink,
        level="WARNING",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
    )

    # 添加文件输出
    logger.add(
        os.path.join(log_dir, "sloop.log"),
        level="DEBUG",
        rotation="10 MB",  # 10MB 轮换
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {thread.name} | {level: <8} | {name}:{function}:{line} - {message}",
        enqueue=True,  # 异步写入，多线程安全
    )


# 导出 logger 实例和配置函数
__all__ = ["logger", "setup_logging"]
