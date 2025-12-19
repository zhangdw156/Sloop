"""
一个模拟的 OpenAI 兼容 API 服务器。
它接受对话请求，并返回一个包含随机字符串的响应。
"""

import random
import string
from dataclasses import dataclass
from typing import Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from uvicorn import Config, Server


@dataclass
class MockMessage:
    """模拟的消息对象"""

    role: str
    content: str


@dataclass
class MockChoice:
    """模拟的选择对象"""

    message: MockMessage


@dataclass
class MockUsage:
    """模拟的使用量统计对象"""

    prompt_tokens: int = 10
    completion_tokens: int = 5
    total_tokens: int = 15


@dataclass
class MockResponse:
    """模拟的响应对象"""

    id: str
    object: str
    created: int
    model: str
    choices: List[MockChoice]
    usage: MockUsage


# Pydantic 模型用于 FastAPI 的请求和响应
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    # 可以添加其他 OpenAI 支持的参数


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]
    usage: Dict[str, int]


class MockOpenAIAPI:
    """
    一个模拟 OpenAI API 的 FastAPI 服务器。
    它在后台运行一个 FastAPI 应用，监听指定的端口。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        """
        初始化。

        Args:
            host (str): 服务器监听的主机地址。
            port (int): 服务器监听的端口。
        """
        self.host = host
        self.port = port
        self.app = FastAPI(title="Mock OpenAI API")
        self.server = None
        self._setup_routes()

    def _setup_routes(self):
        """设置 FastAPI 路由。"""

        @self.app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
        async def create_chat_completion(request: ChatCompletionRequest):
            """
            模拟创建聊天补全。
            根据输入消息的最后一条消息的角色，返回不同的响应。
            """
            # 获取最后一条消息
            last_message = request.messages[-1] if request.messages else None
            response_content = ""

            if last_message and last_message.role == "user":
                # 如果最后是用户消息，返回包含思考和工具调用的内容
                response_content = (
                    "<tool_call>用户想预定餐厅，我需要调用 reserve_restaurant 工具。 found"
                    '{"name": "reserve_restaurant", "arguments": {"name": "美味阁", "time": "19:00"}}<tool_call>'
                )
            elif last_message and last_message.role in ["system", "tool"]:
                # 如果最后是系统或工具消息，返回最终的自然语言总结
                response_content = "“我已经帮您预定好了餐厅。”"
            else:
                # 默认情况，返回一个通用响应
                response_content = "“您好，请问有什么我可以帮您？”"

            # 构造并返回模拟的响应
            return ChatCompletionResponse(
                id="mock_" + "".join(random.choices(string.ascii_lowercase, k=10)),
                object="chat.completion",
                created=1700000000,
                model=request.model,
                choices=[
                    {
                        "message": {"role": "assistant", "content": response_content},
                        "finish_reason": "stop",
                    }
                ],
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )

        # 可以添加更多路由，如 /v1/models 等

    async def start(self):
        """
        启动 FastAPI 服务器。
        """
        config = Config(self.app, host=self.host, port=self.port, log_level="info")
        self.server = Server(config)
        await self.server.serve()

    async def stop(self):
        """
        停止 FastAPI 服务器。
        """
        if self.server:
            self.server.should_exit = True
            await self.server.shutdown()
