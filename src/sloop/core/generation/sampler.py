"""
定义 API 采样器的抽象基类。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class APISampler(ABC):
    """
    API 采样器的抽象基类。
    用于从 API 集合中采样出一个子集。
    """

    @abstractmethod
    def sample(self, apis: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """
        从给定的 API 列表中采样 k 个 API。

        Args:
            apis (List[Dict[str, Any]]): 可供采样的 API 列表。
            k (int): 要采样的 API 数量。

        Returns:
            List[Dict[str, Any]]: 采样得到的 API 列表。
        """
        pass
