import os

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm


class TqdmStream:
    """拦截输出重定向到 tqdm.write，防止进度条断裂"""

    def write(self, msg):
        tqdm.write(str(msg), end="")

    def flush(self):
        pass


def setup_logging():
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "logs",
    )
    os.makedirs(log_dir, exist_ok=True)

    logger.remove()

    # --- 1. Console Handler (Rich + Tqdm) ---
    # 绑定 tqdm 的 console
    tqdm_console = Console(file=TqdmStream(), color_system="auto")

    rich_handler = RichHandler(
        console=tqdm_console,
        show_time=True,
        # [修改点] 设置时间格式为 年-月-日 时:分:秒
        log_time_format="[%Y-%m-%d %H:%M:%S]",
        omit_repeated_times=False,  # 建议关闭，保证每行都有完整时间
        show_level=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )

    logger.add(
        rich_handler,
        level="INFO",
        format="{message}",
    )

    # --- 2. File Handler (全量记录) ---
    logger.add(
        os.path.join(log_dir, "sloop.log"),
        level="DEBUG",
        rotation="10 MB",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {thread.name} | {level: <8} | {name}:{function}:{line} - {message}",
        enqueue=True,
    )


__all__ = ["logger", "setup_logging"]
