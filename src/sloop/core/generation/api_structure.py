"""
定义 API 结构的抽象基类。
"""


from abc import ABC, abstractmethod
from typing import List, Dict, Any


class APIStructure(ABC):
    """
    API 结构的抽象基类。
    用于表示和组织服务/工具的集合。
    """
    @abstractmethod
    def get_all_apis(self) -> List[Dict[str, Any]]:
        """
        获取结构中包含的所有 API。
        
        Returns:
            List[Dict[str, Any]]: API 列表。
        """
        pass
