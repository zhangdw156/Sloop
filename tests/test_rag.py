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
        Mock litellm.embedding，防止真实 API 调用
        """
        # 创建一个假的 embedding 向量 (假设维度 1024)
        fake_vector = [0.1] * 1024
        
        # 模拟 litellm.embedding 的返回值结构
        mock_response = MagicMock()
        mock_item = MagicMock()
        mock_item.embedding = fake_vector
        # 针对 list 输入的返回
        mock_response.data = [mock_item] 
        
        # Patch 掉 sloop.engine.rag 模块里的 litellm
        # 注意：这里要 patch 你的代码里 import litellm 的那个位置
        # 如果你的 rag.py 里是在方法内部 import litellm，则需要 patch 'sys.modules' 或者调整策略
        # 假设 rag.py 头部 import 了 litellm，或者方法内 import
        # 最稳妥的方法是 patch 你的类方法 _get_embedding
        return mocker.patch.object(ToolRetrievalEngine, '_get_embedding', return_value=[fake_vector])

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