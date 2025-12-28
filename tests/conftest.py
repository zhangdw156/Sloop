"""
测试配置和fixtures
"""

import pytest
import json
from pathlib import Path


@pytest.fixture
def sample_apis():
    """示例API定义"""
    return [
        {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {"city": "string", "unit": "string"},
            "category": "weather"
        },
        {
            "name": "search_restaurants",
            "description": "搜索餐厅",
            "parameters": {"city": "string", "cuisine": "string"},
            "category": "travel"
        }
    ]


@pytest.fixture
def sample_services_file(tmp_path, sample_apis):
    """创建临时services.json文件"""
    services_file = tmp_path / "services.json"
    with open(services_file, 'w', encoding='utf-8') as f:
        json.dump(sample_apis, f, ensure_ascii=False, indent=2)
    return str(services_file)


@pytest.fixture
def mock_config(monkeypatch):
    """模拟配置"""
    monkeypatch.setenv("SLOOP_STRONG_API_KEY", "test_key")
    monkeypatch.setenv("SLOOP_STRONG_BASE_URL", "https://api.test.com/v1")
    monkeypatch.setenv("SLOOP_STRONG_MODEL_NAME", "gpt-4o")
