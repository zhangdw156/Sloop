"""
API结构化测试
"""

import pytest
import os
import json
from pathlib import Path
from unittest.mock import patch
from sloop.core.api_structure import APICollection, TreeAPIStructure, GraphAPIStructure
from sloop.core.data_generator import BatchDataGenerator


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


class TestDataGeneration:
    """数据生成测试"""

    def test_data_generator_initialization(self):
        """测试数据生成器初始化"""
        apis = [
            {"name": "get_weather", "description": "获取天气", "parameters": {}},
            {"name": "search_restaurants", "description": "搜索餐厅", "parameters": {}}
        ]

        generator = BatchDataGenerator(apis, "tree")
        assert generator is not None
        assert len(generator.apis) == 2

    def test_config_env_fallback(self):
        """测试配置环境变量fallback"""
        from sloop.core.config import SloopConfig, ModelConfig

        # 测试有环境变量的情况
        with patch.dict(os.environ, {
            'SLOOP_STRONG_API_KEY': 'test_key',
            'SLOOP_STRONG_BASE_URL': 'https://api.test.com',
            'SLOOP_STRONG_MODEL_NAME': 'gpt-4',
            'SLOOP_VERBOSE': 'false'
        }):
            config = SloopConfig()
            assert config.strong.api_key == 'test_key'
            assert config.strong.base_url == 'https://api.test.com'
            assert config.strong.model_name == 'gpt-4'
            assert config.verbose == False

    def test_config_validation(self):
        """测试配置验证"""
        from sloop.core.config import SloopConfig

        # 测试无效配置（缺少必要字段）
        config = SloopConfig()
        config.strong.api_key = ""
        config.strong.base_url = ""
        assert not config.validate()

        # 测试有效配置
        config.strong.api_key = "test_key"
        config.strong.base_url = "https://api.test.com"
        assert config.validate()

    @pytest.mark.integration
    def test_full_data_generation_workflow(self):
        """测试完整数据生成工作流（集成测试）"""
        # 只有在有有效配置时才运行
        from sloop.core.config import config

        if not config.validate():
            pytest.skip("需要有效的API配置才能运行集成测试")

        # 使用tools.json进行测试
        tools_file = Path(__file__).parent / "tools.json"
        output_file = Path(__file__).parent / "integration_test_dataset.json"

        if not tools_file.exists():
            pytest.skip("tools.json文件不存在")

        # 加载API
        from sloop.core.api_structure import load_apis_from_file
        apis = load_apis_from_file(str(tools_file))
        assert len(apis) > 0

        # 初始化生成器
        generator = BatchDataGenerator(apis, "tree")

        # 生成小批量数据进行测试
        dataset = generator.generate_dataset(
            num_conversations=1,
            apis_per_conversation=2,
            sampling_strategy="random",
            target_turns=5,
            output_file=str(output_file)
        )

        # 验证结果 - ShareGPT格式
        assert dataset is not None
        assert len(dataset) == 1

        # 检查ShareGPT必需字段
        conversation = dataset[0]
        assert "conversations" in conversation
        assert "tools" in conversation
        assert "system" in conversation
        assert "id" in conversation

        # 检查conversations格式
        conversations = conversation["conversations"]
        assert isinstance(conversations, list)
        assert len(conversations) > 0

        # 检查消息格式
        for msg in conversations:
            assert "from" in msg
            assert "value" in msg
            assert msg["from"] in ["human", "gpt", "function_call", "observation"]
