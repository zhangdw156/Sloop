"""
测试配置和fixtures
"""

import json
import os

import pytest


@pytest.fixture
def sample_apis():
    """示例API定义"""
    return [
        {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {"city": "string", "unit": "string"},
            "category": "weather",
        },
        {
            "name": "search_restaurants",
            "description": "搜索餐厅",
            "parameters": {"city": "string", "cuisine": "string"},
            "category": "travel",
        },
    ]


@pytest.fixture
def sample_services_file(sample_apis):
    """创建临时sample_apis.json文件在tests目录下"""
    test_dir = os.path.dirname(__file__)  # tests目录
    services_file = os.path.join(test_dir, "sample_services.json")

    with open(services_file, "w", encoding="utf-8") as f:
        json.dump(sample_apis, f, ensure_ascii=False, indent=2)

    yield services_file

    # 测试结束后清理文件
    if os.path.exists(services_file):
        os.remove(services_file)


@pytest.fixture
def mock_config(monkeypatch):
    """模拟配置"""
    monkeypatch.setenv("SLOOP_STRONG_API_KEY", "test_key")
    monkeypatch.setenv("SLOOP_STRONG_BASE_URL", "https://api.test.com/v1")
    monkeypatch.setenv("SLOOP_STRONG_MODEL_NAME", "gpt-4o")
