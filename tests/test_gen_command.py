"""
测试 sloop 的 `gen` 命令。
"""

import json
import os

from typer.testing import CliRunner

from sloop.cli.main import app


def test_gen_command_with_mock_server(mock_api_server):
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

    # 从 fixture 获取服务器信息
    port = mock_api_server.port
    base_url = f"http://127.0.0.1:{port}/v1"  # 确保包含 /v1 路径

    # 调用 `gen` 命令，并显式传递环境变量
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
        env={
            "SLOOP_STRONG_API_KEY": "test_key",
            "SLOOP_STRONG_BASE_URL": base_url,
            "SLOOP_WEAK_API_KEY": "test_weak_key",
            "SLOOP_WEAK_BASE_URL": base_url,
        },
    )

    # 检查命令是否成功执行
    assert result.exit_code == 0, (
        f"命令执行失败，返回码: {result.exit_code}, 错误: {result.output}"
    )

    # 检查输出文件是否被创建
    assert os.path.exists(test_output_file), "输出文件未被创建"

    # 读取并验证输出文件内容
    with open(test_output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 验证数据结构
    assert isinstance(data, list), "输出数据应为列表"
    assert len(data) > 0, "输出数据不应为空"
    assert "conversation" in data[0], "对话数据中应包含 'conversation' 字段"
    assert "label" in data[0], "对话数据中应包含 'label' 字段"

    # 清理测试文件
    # os.remove(test_output_file)
