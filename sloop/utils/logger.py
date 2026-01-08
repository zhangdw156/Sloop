import os
import sys

from loguru import logger
from tqdm import tqdm
from rich.logging import RichHandler
from rich.console import Console

# 1. 定义一个伪造的文件流，把 write 调用转发给 tqdm.write
class TqdmStream:
    """
    一个适配器类，充当文件流对象。
    它拦截写入操作并将其重定向到 tqdm.write，
    从而确保 Rich 的输出不会破坏 tqdm 进度条。
    """
    def write(self, msg):
        # end="" 是因为 Rich 已经处理了换行，tqdm.write 默认也会加换行
        # msg 包含了 Rich 生成的 ANSI 颜色代码，tqdm 能很好地处理它们
        tqdm.write(str(msg), end="")

    def flush(self):
        # 接口兼容性需要，实际不需要做 flush，因为 tqdm.write 是即时的
        pass

def setup_logging():
    """
    设置日志配置
    
    Console: 使用 RichHandler + tqdm.write，只显示 WARNING/ERROR
    File: 记录所有 DEBUG/INFO，保留详细的线程和路径信息
    """
    # 确保 logs 目录存在
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "logs",
    )
    os.makedirs(log_dir, exist_ok=True)

    # 移除默认 handler
    logger.remove()

    # --- Console Handler (Rich + Tqdm) ---
    
    # 创建一个绑定了 tqdm.write 的 Rich Console
    # width=None 让它自适应终端宽度，color_system="auto" 自动检测颜色支持
    tqdm_console = Console(file=TqdmStream(), color_system="auto")

    # 配置 RichHandler
    # rich_tracebacks=True 是核心功能，让报错堆栈变漂亮
    rich_handler = RichHandler(
        console=tqdm_console,
        show_time=True,       # Rich 自带时间显示
        show_level=True,      # Rich 自带等级显示
        show_path=False,      # 控制台通常不需要显示长路径，保持清爽
        markup=True,          # 允许在日志里写 [bold red]...[/] 这种标记
        rich_tracebacks=True  # 启用超美观的异常堆栈打印
    )

    logger.add(
        rich_handler,
        level="WARNING",  # 按照你的要求，控制台只看警告和错误
        # 注意：这里 format 设为 "{message}"，因为 RichHandler 自己会处理 时间/等级 的排版
        # 我们只需要把纯消息传给它即可。如果这里加了 {time}，日志里就会出现两遍时间。
        format="{message}", 
    )

    # --- File Handler (详细记录) ---
    logger.add(
        os.path.join(log_dir, "sloop.log"),
        level="DEBUG",
        rotation="10 MB",
        # 文件里我们不需要 Rich 的花哨格式，保留最原始详细的 Loguru 格式
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {thread.name} | {level: <8} | {name}:{function}:{line} - {message}",
        enqueue=True,
    )

# 导出
__all__ = ["logger", "setup_logging"]