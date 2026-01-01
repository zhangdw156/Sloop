"""
向量检索引擎

实现基于 FAISS 的工具向量检索，用于 RAG 增强的采样逻辑。
"""

import json
import os
from pathlib import Path
from typing import List

import faiss
import numpy as np
from tqdm import tqdm

from sloop.config import get_settings
from sloop.models import ToolDefinition
from sloop.utils.logger import logger


class ToolRetrievalEngine:
    """
    工具向量检索引擎

    使用 FAISS 构建工具向量索引，支持语义搜索相似工具。
    """

    def __init__(self, cache_dir: str = ".cache"):
        """
        初始化检索引擎

        参数:
            cache_dir: 缓存目录路径，相对于当前工作目录
        """
        self.cache_dir = Path(cache_dir)
        self.index_path = self.cache_dir / "tool_index.faiss"
        self.names_path = self.cache_dir / "tool_names.json"

        # 确保缓存目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 初始化属性
        self.index = None
        self.tool_names = []

        # 尝试加载现有索引
        self._load_index()

        # 获取配置
        self.settings = get_settings()

    def _load_index(self):
        """加载现有索引"""
        if self.index_path.exists() and self.names_path.exists():
            try:
                # 加载 FAISS 索引
                self.index = faiss.read_index(str(self.index_path))

                # 加载工具名称映射
                with open(self.names_path, 'r', encoding='utf-8') as f:
                    self.tool_names = json.load(f)

                logger.info(f"Loaded index successfully: {len(self.tool_names)} tools")

            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
                self.index = None
                self.tool_names = []
        else:
            logger.info("No existing index files found")

    def _save_index(self):
        """保存索引到磁盘"""
        if self.index is not None and self.tool_names:
            try:
                # 保存 FAISS 索引
                faiss.write_index(self.index, str(self.index_path))

                # 保存工具名称映射
                with open(self.names_path, 'w', encoding='utf-8') as f:
                    json.dump(self.tool_names, f, ensure_ascii=False, indent=2)

                logger.info(f"Index saved successfully: {self.index_path}, {self.names_path}")

            except Exception as e:
                logger.error(f"Failed to save index: {e}")

    def _get_embedding(self, text: str | List[str]) -> List[float] | List[List[float]]:
        """
        获取文本的向量表示

        参数:
            text: 输入文本或文本列表

        返回:
            向量列表或向量列表的列表
        """
        import litellm

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = litellm.embedding(
                    model=f"{self.settings.embedding_provider}/{self.settings.embedding_model}",
                    input=text,
                    api_key=self.settings.embedding_api_key,
                    api_base=self.settings.embedding_base_url,
                    encoding_format="float",
                )

                if response and response.data:
                    if isinstance(text, str):
                        # 单条输入
                        if len(response.data) > 0:
                            item = response.data[0]
                            if hasattr(item, 'embedding'):
                                return item.embedding
                            elif isinstance(item, dict) and 'embedding' in item:
                                return item['embedding']
                            else:
                                # 假设 item 就是向量列表
                                return item
                    else:
                        # 批量输入
                        embeddings = []
                        for item in response.data:
                            if hasattr(item, 'embedding'):
                                embeddings.append(item.embedding)
                            elif isinstance(item, dict) and 'embedding' in item:
                                embeddings.append(item['embedding'])
                            else:
                                # 假设 item 就是向量列表
                                embeddings.append(item)
                        return embeddings

            except Exception as e:
                logger.warning(f"Embedding call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    continue

        # 失败时抛出异常
        raise RuntimeError("Embedding call failed after all retries")

    def build(self, tools: List[ToolDefinition], force: bool = False):
        """
        构建工具向量索引

        参数:
            tools: 工具定义列表
            force: 是否强制重新构建
        """
        # 检查是否需要跳过构建
        if not force and self.index is not None and self.tool_names:
            logger.info("Index already exists, skipping build (use force=True to rebuild)")
            return

        logger.info(f"Starting to build index: {len(tools)} tools")

        # 准备数据
        texts = []
        self.tool_names = []

        for tool in tools:
            # 构造语义文本
            params_str = json.dumps(tool.parameters.model_dump(), ensure_ascii=False)
            text = f"name: {tool.name} description: {tool.description} params: {params_str}"
            texts.append(text)
            self.tool_names.append(tool.name)

        logger.info("Generating embeddings...")

        # 批量生成向量（分批处理，避免 API 限制）
        batch_size = 10
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]

            # 批量调用 embedding API
            batch_embeddings = self._get_embedding(batch_texts)
            all_embeddings.extend(batch_embeddings)

        # 转换为 numpy 数组
        embeddings = np.array(all_embeddings, dtype=np.float32)

        logger.info(f"Embeddings shape: {embeddings.shape}")

        # 构建 FAISS 索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        logger.info(f"Index build completed: {len(self.tool_names)} tools")

        # 保存索引
        self._save_index()

    def search(self, query_tool: ToolDefinition, top_k: int = 10) -> List[str]:
        """
        搜索相似的工具

        参数:
            query_tool: 查询工具定义
            top_k: 返回的相似工具数量

        返回:
            相似工具名称列表
        """
        if self.index is None or not self.tool_names:
            logger.error("Index not built, cannot search")
            return []

        # 构造查询文本
        params_str = json.dumps(query_tool.parameters.model_dump(), ensure_ascii=False)
        query_text = f"name: {query_tool.name} description: {query_tool.description} params: {params_str}"

        # 获取查询向量
        query_embedding = self._get_embedding(query_text)
        query_vector = np.array([query_embedding], dtype=np.float32)

        # 搜索相似向量
        distances, indices = self.index.search(query_vector, min(top_k, len(self.tool_names)))

        # 返回工具名称
        results = []
        for idx in indices[0]:
            if idx < len(self.tool_names):
                results.append(self.tool_names[idx])

        return results


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("🔍 ToolRetrievalEngine 测试")
    print("=" * 50)

    # 创建模拟工具
    mock_tools = [
        ToolDefinition(
            name="get_weather",
            description="获取指定城市的天气信息",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "date": {"type": "string", "description": "日期"},
                },
                "required": ["city"],
            },
        ),
        ToolDefinition(
            name="search_restaurants",
            description="搜索指定城市的餐厅",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "cuisine": {"type": "string", "description": "菜系类型"},
                    "price_range": {"type": "string", "description": "价格范围"},
                },
                "required": ["city"],
            },
        ),
        ToolDefinition(
            name="book_hotel",
            description="预订酒店房间",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "check_in": {"type": "string", "description": "入住日期"},
                    "check_out": {"type": "string", "description": "退房日期"},
                    "guests": {"type": "integer", "description": "入住人数"},
                },
                "required": ["city", "check_in", "check_out"],
            },
        ),
        ToolDefinition(
            name="send_email",
            description="发送电子邮件",
            parameters={
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "收件人邮箱"},
                    "subject": {"type": "string", "description": "邮件主题"},
                    "body": {"type": "string", "description": "邮件正文"},
                },
                "required": ["to", "subject", "body"],
            },
        ),
        ToolDefinition(
            name="calculate_distance",
            description="计算两地之间的距离",
            parameters={
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "起点"},
                    "destination": {"type": "string", "description": "终点"},
                    "mode": {"type": "string", "description": "出行方式", "enum": ["driving", "walking", "transit"]},
                },
                "required": ["origin", "destination"],
            },
        ),
    ]

    print(f"📋 测试数据: {len(mock_tools)} 个工具")
    for tool in mock_tools:
        print(f"  - {tool.name}: {tool.description}")

    # 初始化引擎
    print("\n🔧 初始化 ToolRetrievalEngine...")
    engine = ToolRetrievalEngine()

    # 构建索引
    print("\n🏗️ 构建索引...")
    engine.build(mock_tools, force=True)

    # 测试搜索
    print("\n🔍 测试搜索...")

    # 使用第一个工具作为查询
    query_tool = mock_tools[0]  # get_weather
    print(f"📝 查询工具: {query_tool.name} - {query_tool.description}")

    results = engine.search(query_tool, top_k=3)
    print(f"🎯 相似工具 (Top-3): {results}")

    # 使用另一个工具测试
    query_tool2 = mock_tools[3]  # send_email
    print(f"\n📝 查询工具: {query_tool2.name} - {query_tool2.description}")

    results2 = engine.search(query_tool2, top_k=3)
    print(f"🎯 相似工具 (Top-3): {results2}")

    print("\n✅ ToolRetrievalEngine 测试完成！")
    print(f"📁 检查缓存文件: {engine.cache_dir}")
