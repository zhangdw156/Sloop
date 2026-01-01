import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import numpy as np

from sloop.engine.rag import ToolRetrievalEngine
from sloop.models import ToolDefinition

class TestToolRetrievalEngine:
    """ToolRetrievalEngine 测试类"""

    @pytest.fixture
    def temp_cache_dir(self):
        """创建临时缓存目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def mock_tools(self):
        """创建模拟工具 (保持不变)"""
        return [
            ToolDefinition(
                name="get_weather",
                description="获取天气",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            ),
            ToolDefinition(
                name="book_hotel",
                description="预订酒店",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            ),
        ]

    @pytest.fixture
    def mock_embedding(self, mocker):
        """
        Mock _get_embedding 方法，防止真实 API 调用
        """
        # 创建一个假的 embedding 向量 (假设维度 1024)
        fake_vector = [0.1] * 1024

        def mock_get_embedding(text):
            """Mock _get_embedding 方法"""
            if isinstance(text, str):
                # 单条输入，返回单个向量
                return fake_vector
            else:
                # 批量输入，返回向量列表
                return [fake_vector] * len(text)

        return mocker.patch.object(ToolRetrievalEngine, '_get_embedding', side_effect=mock_get_embedding)

    def test_build_and_search(self, temp_cache_dir, mock_tools, mock_embedding):
        """测试构建和搜索 (使用 Mock)"""
        engine = ToolRetrievalEngine(cache_dir=str(temp_cache_dir))

        # 构建索引 (此时 _get_embedding 被 mock，不会真的联网)
        engine.build(mock_tools, force=True)

        assert engine.index is not None
        assert len(engine.tool_names) == 2
        
        # 验证 Mock 被调用了 (说明逻辑走通了)
        assert mock_embedding.called

        # 测试搜索
        query_tool = mock_tools[0]
        results = engine.search(query_tool, top_k=1)
        
        assert len(results) == 1
        assert results[0] in [t.name for t in mock_tools]

    def test_build_idempotent(self, temp_cache_dir, mock_tools, mock_embedding):
        """测试幂等性 (验证是否真的跳过了 API 调用)"""
        engine = ToolRetrievalEngine(cache_dir=str(temp_cache_dir))

        # 第一次构建
        engine.build(mock_tools, force=True)
        # 重置 mock 的调用计数
        mock_embedding.reset_mock()

        # 第二次构建（force=False）
        engine.build(mock_tools, force=False)

        # 关键断言：验证 _get_embedding 完全没有被调用！
        # 这证明代码真的跳过了构建步骤，而不仅仅是结果碰巧一样
        mock_embedding.assert_not_called()

    def test_build_force_rebuild(self, temp_cache_dir, mock_tools, mock_embedding):
        """测试强制重建"""
        engine = ToolRetrievalEngine(cache_dir=str(temp_cache_dir))

        engine.build(mock_tools, force=True)
        mock_embedding.reset_mock()

        # 强制重建
        engine.build(mock_tools, force=True)

        # 关键断言：验证 _get_embedding 又被调用了
        assert mock_embedding.called
