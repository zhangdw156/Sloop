"""
Sloop：基于下推自动机（PDA）与状态模拟的 Agent 数据合成引擎，用于生成含思维链（CoT）的复杂工具调用训练集。
"""

__version__ = "0.1.2"

# 导出主要组件
from sloop import config, engine, models, utils
