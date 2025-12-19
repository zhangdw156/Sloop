"""
提供一个随机采样的实现。
"""
import random
from typing import List, Dict, Any

from ..sampler import APISampler

class RandomAPISampler(APISampler):
    """
    一个随机 API 采样器的实现。
    """
    def sample(
        self, 
        apis: List[Dict[str, Any]], 
        k: int
    ) -> List[Dict[str, Any]]:
        """
        从给定的 API 列表中随机采样 k 个 API。
        
        Args:
            apis (List[Dict[str, Any]]): 可供采样的 API 列表。
            k (int): 要采样的 API 数量。
            
        Returns:
            List[Dict[str, Any]]: 采样得到的 API 列表。
        """
        return random.sample(apis, min(k, len(apis)))
