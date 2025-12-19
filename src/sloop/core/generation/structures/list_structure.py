"""
提供一个简单的 API 列表实现。
"""
from typing import List, Dict, Any

from ..api_structure import APIStructure

class ListAPIStructure(APIStructure):
    """
    一个简单的 API 列表结构实现。
    """
    def __init__(self, apis: List[Dict[str, Any]]):
        """
        初始化。
        
        Args:
            apis (List[Dict[str, Any]]): API 列表。
        """
        self._apis = apis

    def get_all_apis(self) -> List[Dict[str, Any]]:
        """
        获取所有 API。
        
        Returns:
            List[Dict[str, Any]]: API 列表。
        """
        return self._apis
