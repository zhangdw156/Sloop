"""
测试日志管理模块

为测试文件提供统一的日志管理。
"""

import logging
from pathlib import Path


def get_test_logger(module_name: str) -> logging.Logger:
    """
    获取测试日志器

    参数:
        module_name: 模块名称，用于生成日志文件名

    返回:
        配置好的日志器实例
    """
    # 移除 .py 后缀
    if module_name.endswith(".py"):
        module_name = module_name[:-3]

    # 创建 logger
    logger_name = f"test_{module_name}"
    logger = logging.getLogger(logger_name)

    # 如果已经配置过，直接返回
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # 确保 logs/tests 目录存在
    test_log_dir = Path(__file__).parent.parent / "logs" / "tests"
    test_log_dir.mkdir(parents=True, exist_ok=True)

    # 文件handler
    log_file = test_log_dir / f"{module_name}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    # 添加handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_current_test_logger() -> logging.Logger:
    """
    获取当前测试文件的日志器

    自动从调用栈中提取测试文件名
    """
    import inspect

    # 获取调用栈
    frame = inspect.currentframe()
    try:
        # 向上查找测试文件
        while frame:
            filename = frame.f_code.co_filename
            if "test_" in filename and filename.endswith(".py"):
                # 提取文件名（不含路径）
                module_name = Path(filename).stem
                return get_test_logger(module_name)
            frame = frame.f_back
    finally:
        del frame

    # 默认返回通用测试 logger
    return get_test_logger("test")
