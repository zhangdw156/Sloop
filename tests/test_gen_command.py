"""
测试 sloop 的 `gen` 命令。
"""
import json
import os
import asyncio
import pytest
from typer.testing import CliRunner
from sloop.cli.main import app
from tests.mock_api import MockOpenAIAPI


@pytest.fixture(scope="module")
async def mock_api_server():
    """启动一个模块级别的 mock API 服务器 fixture。"""
    # 选择一个随机端口
    import random
    port = random.randint(8000, 9000)
    
    # 设置测试环境变量，指向本地运行的 mock 服务
    os.environ["SLOOP_STRONG_API_KEY"] = "test_key"
    os.environ["SLOOP_STRONG_BASE_URL"] = f"http://127.0.0.1:{port}"
    # 为了通过 SloopConfig 的验证，弱模型的环境变量也需要设置，同样指向 mock 服务或一个占位符
    os.environ["SLOOP_WEAK_API_KEY"] = "test_weak_key"
    os.environ["SLOOP_WEAK_BASE_URL"] = f"http://127.0.0.1:{port}"

    # 创建 mock API 服务器实例
    server = MockOpenAIAPI(port=port)
    # 在后台启动服务器
    server_task = asyncio.create_task(server.start())
    
    # 等待服务器启动（这里可以添加更健壮的健康检查）
    await asyncio.sleep(0.1)
    
    # 生成器在测试后执行清理
    yield server
    
    # 停止服务器
    await server.stop()
    # 等待服务器任务完成
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass


async def test_gen_command_with_mock_server(mock_api_server):
    """
    测试 `gen` 命令在使用本地 mock API 服务器时是否能成功执行。
    """
    # 创建一个 CLI Runner
    runner = CliRunner()

    # 准备临时的输出文件路径
    test_output_file = "tests/test_output.json"
    # 确保输出文件不存在
    if os.path.exists(test_output_file):
        os.remove(test_output_file)

    # 调用 `gen` 命令
    result = runner.invoke(
        app,
        [
            "gen",
            "--services",
            "tests/services.json",
            "--output",
            test_output_file,
            "--agent-config",
            "configs/default_agents.yaml",
        ],
    )

    # 检查命令是否成功执行
    assert result.exit_code == 0, (
        f"命令执行失败，返回码: {result.exit_code}, 错误: {result.output}"
    )

    # 检查输出文件是否被创建
    assert os.path.exists(test_output_file), "输出文件未被创建"

    # 读取并验证输出文件内容
    with open(test_output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 验证数据结构
    assert isinstance(data, list), "输出数据应为列表"
    assert len(data) > 0, "输出数据不应为空"
    assert "conversation" in data[0], "对话数据中应包含 'conversation' 字段"
    assert "label" in data[0], "对话数据中应包含 'label' 字段"

    # 清理测试文件
    os.remove(test_output_file)
