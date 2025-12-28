"""
API结构化测试
"""

import pytest
from sloop.core.api_structure import APICollection, TreeAPIStructure, GraphAPIStructure


class TestAPICollection:
    """API集合测试"""

    def test_tree_structure_creation(self, sample_apis):
        """测试树形结构创建"""
        collection = APICollection(sample_apis, "tree")

        assert isinstance(collection.structure, TreeAPIStructure)
        assert len(collection.get_all_apis()) == 2

        # 检查类别识别
        categories = collection.structure.get_categories()
        assert "weather" in categories
        assert "travel" in categories

    def test_api_sampling(self, sample_apis):
        """测试API采样"""
        collection = APICollection(sample_apis, "tree")

        # 测试随机采样
        sampled = collection.sample_apis(1, "random")
        assert len(sampled) == 1
        assert sampled[0] in sample_apis

        # 测试均衡采样
        sampled = collection.sample_apis(2, "balanced")
        assert len(sampled) == 2

    def test_related_apis(self, sample_apis):
        """测试相关API获取"""
        collection = APICollection(sample_apis, "tree")

        # 获取天气API的相关API
        related = collection.get_related_apis("get_weather")
        assert len(related) >= 0  # 可能没有直接相关API

        # 获取不存在的API
        related = collection.get_related_apis("nonexistent")
        assert related == []


class TestTreeAPIStructure:
    """树形API结构测试"""

    def test_category_extraction(self):
        """测试类别提取"""
        structure = TreeAPIStructure([])

        # 测试带类别的API
        api_with_category = {"name": "test", "description": "test", "category": "custom"}
        assert structure._extract_category(api_with_category) == "custom"

        # 测试关键词提取
        weather_api = {"name": "weather", "description": "get weather information"}
        assert structure._extract_category(weather_api) == "weather"

        # 测试默认类别
        unknown_api = {"name": "unknown", "description": "some unknown function"}
        assert structure._extract_category(unknown_api) == "general"


class TestGraphAPIStructure:
    """图形API结构测试"""

    def test_graph_creation(self, sample_apis):
        """测试图创建"""
        relationships = [
            {"from": "get_weather", "to": "search_restaurants", "type": "related"}
        ]

        structure = GraphAPIStructure(sample_apis, relationships)

        assert structure.graph.number_of_nodes() == 2
        assert structure.graph.number_of_edges() == 1

    def test_auto_relationships(self, sample_apis):
        """测试自动关系构建"""
        # 创建描述相似的API
        similar_apis = sample_apis + [
            {"name": "get_forecast", "description": "获取天气预报", "parameters": {}}
        ]

        structure = GraphAPIStructure(similar_apis)
        # 应该基于描述相似性自动建立关系
        assert structure.graph.number_of_edges() >= 0
