# FILE: sloop/services/embedding.py
from typing import List, Union

import numpy as np
from openai import OpenAI, APIError

from sloop.configs import env_config
from sloop.utils.logger import logger
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingService:
    def __init__(self):
        # 从 env_config 获取配置
        self.base_url = env_config.get("EMBEDDING_MODEL_BASE_URL")
        self.api_key = env_config.get("EMBEDDING_MODEL_API_KEY")
        self.model_name = env_config.get("EMBEDDING_MODEL_NAME", "bge-m3")

        if not self.base_url or not self.api_key:
            logger.error("Missing embedding configuration in .env")
            raise ValueError("Missing embedding configuration")

        # 使用 OpenAI SDK 初始化客户端
        # 注意：通常 base_url 应该以 /v1 结尾（取决于你的服务提供商配置），
        # SDK 会自动处理 /embeddings 路径后缀。
        try:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                # 如果你的内网服务有自签名证书问题，可以使用 http_client 参数传入 verify=False 的 httpx client
                # 但标准用法通常不需要
            )
            logger.info(f"Embedding Service initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Embedding client: {e}")
            raise e

    def get_embedding(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """获取文本向量，支持单条或批量"""
        # OpenAI SDK 的 embedding 输入如果是一维列表，返回也是列表对象
        # 需要处理单条和批量的情况
        if not self.client:
             raise RuntimeError("Embedding client not initialized")

        # 确保输入不为空
        if not text:
            return [] if isinstance(text, list) else None

        try:
            # 这里的 text 可以是 string 或 list of strings
            response = self.client.embeddings.create(
                input=text,
                model=self.model_name
            )
            
            # 提取向量数据
            # response.data 是一个 Embedding 对象列表，按 index 排序
            # 为了保险，我们按 index 排序提取 embedding
            sorted_data = sorted(response.data, key=lambda x: x.index)
            embeddings = [item.embedding for item in sorted_data]

            # 如果输入是单条字符串，返回单条向量
            if isinstance(text, str):
                return embeddings[0]
            
            return embeddings

        except Exception as e:
            logger.error(f"Embedding API call failed: {e}")
            raise e

    def compute_similarity(self, vec_a, vec_b):
        """计算余弦相似度"""
        vec_a = np.array(vec_a)
        vec_b = np.array(vec_b)
        if vec_a.ndim == 1: vec_a = vec_a.reshape(1, -1)
        if vec_b.ndim == 1: vec_b = vec_b.reshape(1, -1)

        return cosine_similarity(vec_a, vec_b)[0][0]