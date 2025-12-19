"""
Pytest 配置文件，用于集中管理测试 fixture。
"""

import asyncio
import threading
import time

import pytest
from uvicorn import Config, Server

from tests.mock_api import MockOpenAIAPI


@pytest.fixture(scope="module")
def mock_api_server():
    """启动一个模块级别的 mock API 服务器 fixture，并在独立线程中运行。"""
    # 选择一个随机端口
    port = 8099  # 为了调试方便，暂时使用固定端口
    host = "127.0.0.1"

    # 创建 mock API 服务器实例
    server = MockOpenAIAPI(host=host, port=port)

    # 定义在独立线程中运行服务器的函数
    def run_server():
        # 为子线程创建新的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            config = Config(server.app, host=host, port=port, log_level="info")
            server_instance = Server(config)
            loop.run_until_complete(server_instance.serve())
        except Exception as e:
            print(f"Server thread error: {e}")
        finally:
            loop.close()

    # 在独立线程中启动服务器
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # 等待服务器启动（这里可以添加更健壮的健康检查）
    # 简单等待 0.5 秒
    time.sleep(0.5)

    # 通过属性暴露端口，供测试使用
    server.port = port

    # 生成器在测试后执行清理
    return server

    # 停止服务器（如果需要）
    # 由于是 daemon 线程，通常不需要显式停止
