"""
具体的 API 结构实现，用于表示单个 API。
"""

from typing import Dict

from .api_structure import APIStructure


class API(APIStructure):
    """
    单个 API 的具体实现。
    """

    def __init__(self, name: str, description: str, parameters: Dict[str, str]):
        """
        初始化。

        Args:
            name (str): API 名称。
            description (str): API 描述。
            parameters (Dict[str, str]): 参数字典，键为参数名，值为参数类型。
        """
        self.name = name
        self.description = description
        self.parameters = parameters

    def get_all_apis(self) -> list:
        """
        获取该 API 的信息。

        Returns:
            list: 包含单个 API 信息的列表。
        """
        return [self.__dict__]
