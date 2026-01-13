from openai import AsyncOpenAI
from dataclasses import dataclass
from typing import List

@dataclass
class EmbeddingResponse:
    embeddings: List[List[float]]

class SafeOpenAIEmbedding:
    """
    一个更纯净的 Embedding 客户端，防止发送不支持的 'dimensions' 参数。
    模拟 AgentScope 的调用接口。
    """
    def __init__(self, api_key, model_name, base_url):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    async def __call__(self, inputs: List[str]) -> EmbeddingResponse:
        # 这里直接调用 openai 原生接口，且严格控制参数，不传 dimensions
        response = await self.client.embeddings.create(
            input=inputs,
            model=self.model_name,
            encoding_format="float"
        )
        # 提取向量，保持与 AgentScope 返回格式一致
        vectors = [item.embedding for item in response.data]
        return EmbeddingResponse(embeddings=vectors)