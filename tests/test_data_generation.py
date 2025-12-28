"""
专门的数据生成测试
测试完整的CLI工作流和数据格式
"""

import pytest
import json
import subprocess
import os
from pathlib import Path
from unittest.mock import patch


class TestDataGenerationCLI:
    """CLI数据生成测试"""

    def test_gen_command_help(self):
        """测试gen命令帮助信息"""
        result = subprocess.run(
            ["uv", "run", "sloop", "gen", "--help"],
            capture_output=True,
            text=True,
            cwd="/Users/zhangdw/work/lenovo/Sloop"
        )
        assert result.returncode == 0
        assert "--num-conversations" in result.stdout
        assert "--yes" in result.stdout

    def test_validate_command_help(self):
        """测试validate命令帮助信息"""
        result = subprocess.run(
            ["uv", "run", "sloop", "validate", "--help"],
            capture_output=True,
            text=True,
            cwd="/Users/zhangdw/work/lenovo/Sloop"
        )
        assert result.returncode == 0
        assert "--dataset" in result.stdout

    @pytest.mark.integration
    def test_full_cli_workflow(self):
        """测试完整的CLI工作流"""
        # 只有在有有效配置时才运行
        from sloop.core.config import config
        if not config.validate():
            pytest.skip("需要有效的API配置才能运行集成测试")

        # 设置测试环境
        test_dir = Path("/Users/zhangdw/work/lenovo/Sloop/tests/data")
        output_file = test_dir / "cli_test_dataset.json"

        # 清理旧文件
        if output_file.exists():
            output_file.unlink()

        try:
            # 执行gen命令
            result = subprocess.run(
                [
                    "uv", "run", "sloop", "gen",
                    "--num-conversations", "1",
                    "--yes",
                    "--output", str(output_file)
                ],
                capture_output=True,
                text=True,
                cwd="/Users/zhangdw/work/lenovo/Sloop"
            )

            # 检查命令执行成功
            assert result.returncode == 0
            assert "生成完成" in result.stdout
            assert output_file.exists()

            # 验证生成的文件格式
            with open(output_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)

            assert isinstance(dataset, list)
            assert len(dataset) == 1

            conversation = dataset[0]
            assert "conversations" in conversation
            assert "tools" in conversation
            assert "system" in conversation
            assert "id" in conversation

            # 执行validate命令
            validate_result = subprocess.run(
                [
                    "uv", "run", "sloop", "validate",
                    "--dataset", str(output_file)
                ],
                capture_output=True,
                text=True,
                cwd="/Users/zhangdw/work/lenovo/Sloop"
            )

            # 检查验证成功
            assert validate_result.returncode == 0
            assert "数据集质量良好" in validate_result.stdout

        finally:
            # 清理测试文件
            if output_file.exists():
                output_file.unlink()

    def test_gen_without_confirmation_fails(self):
        """测试不使用--yes参数时命令失败"""
        result = subprocess.run(
            [
                "uv", "run", "sloop", "gen",
                "--num-conversations", "1",
                "--output", "test.json"
            ],
            capture_output=True,
            text=True,
            cwd="/Users/zhangdw/work/lenovo/Sloop",
            timeout=5  # 5秒超时，因为会等待用户输入
        )

        # 命令应该超时或失败，因为等待用户输入
        assert result.returncode != 0 or "开始生成数据集?" in result.stdout


class TestShareGPTFormat:
    """ShareGPT格式测试"""

    def test_sharegpt_format_validation(self):
        """测试ShareGPT格式验证"""
        # 构造一个有效的ShareGPT格式数据
        valid_data = {
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi there"},
                {"from": "function_call", "value": '{"name": "test", "arguments": {}}'},
                {"from": "observation", "value": '{"result": "ok"}'}
            ],
            "tools": '[{"name": "test", "description": "test", "parameters": {}}]',
            "system": "You are a helpful assistant",
            "id": "test_001"
        }

        # 检查所有必需字段存在
        assert "conversations" in valid_data
        assert "tools" in valid_data
        assert "system" in valid_data
        assert "id" in valid_data

        # 检查conversations格式
        conversations = valid_data["conversations"]
        assert isinstance(conversations, list)
        assert len(conversations) == 4

        # 检查每条消息的格式
        for msg in conversations:
            assert "from" in msg
            assert "value" in msg
            assert msg["from"] in ["human", "gpt", "function_call", "observation"]

    def test_tools_format_validation(self):
        """测试tools字段格式"""
        import json

        tools_str = '[{"name": "test", "description": "test", "parameters": {}}]'
        tools_data = json.loads(tools_str)

        assert isinstance(tools_data, list)
        assert len(tools_data) == 1
        assert "name" in tools_data[0]
        assert "description" in tools_data[0]
        assert "parameters" in tools_data[0]
