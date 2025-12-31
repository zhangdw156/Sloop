from loguru import logger
import os

# 确保 logs 目录存在
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
os.makedirs(log_dir, exist_ok=True)

# 移除默认的 loguru handler
logger.remove()

# 添加控制台输出
logger.add(
    "sys.stderr",
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
)

# 添加文件输出，支持轮换和压缩
logger.add(
    os.path.join(log_dir, "file_{time:YYYY-MM-DD}.log"),
    level="DEBUG",
    rotation="1 day",  # 每天轮换
    compression="zip",  # 压缩旧日志文件
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    enqueue=True,  # 异步写入
    retention="7 days", # 保留7天日志
)

# 可以根据需要添加更多的配置，例如不同的日志级别、过滤器等

# 导出 logger 实例
__all__ = ["logger"]
