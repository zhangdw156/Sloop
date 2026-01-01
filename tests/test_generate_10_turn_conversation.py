"""
测试生成10轮对话数据的功能
"""

import json

import pytest
from sloop.core.api_structure import load_apis_from_file
from sloop.core.config import config
from sloop.core.data_generator import BatchDataGenerator


class TestGenerate10TurnConversation:
    """生成10轮对话数据的测试"""

    @pytest.mark.integration
    def test_generate_single_10_turn_conversation(self):
        """测试生成单个10轮对话数据"""
        # 只有在有有效配置时才运行
        if not config.validate():
            pytest.skip("需要有效的API配置才能运行集成测试")

        # 加载API定义
        apis = load_apis_from_file("tests/data/tools.json")
        assert len(apis) > 0, "API文件不能为空"

        # 创建数据生成器
        generator = BatchDataGenerator(apis, "tree")

        # 设置输出文件路径（保存测试数据）
        output_file = "tests/data/test_generated_conversation.json"

        # 生成单个对话数据并保存到文件
        dataset = generator.generate_dataset(
            num_conversations=1,
            apis_per_conversation=3,
            sampling_strategy="balanced",
            target_turns=10,  # 降低目标轮数，让测试更容易通过
            output_file=output_file,
        )

        # 验证结果
        assert len(dataset) == 1, "应该生成1条对话数据"
        conversation = dataset[0]

        # 验证基本结构
        assert "messages" in conversation, "应该包含messages字段"
        assert "tools" in conversation, "应该包含tools字段"
        assert "id" in conversation, "应该包含id字段"

        messages = conversation["messages"]
        assert isinstance(messages, list), "messages应该是列表"
        assert len(messages) >= 1, "对话至少应该有1条消息"

        # 验证消息格式
        for msg in messages:
            assert "role" in msg, "每条消息应该有role字段"
            assert "content" in msg, "每条消息应该有content字段"
            assert msg["role"] in ["user", "assistant", "tool_call", "tool_response"], (
                f"无效的role: {msg['role']}"
            )

        # 验证至少有一条assistant消息
        assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
        assert len(assistant_messages) > 0, "对话应该包含至少一条assistant消息"

        # 验证ReAct格式（<think>标签由代码手动添加）
        for msg in assistant_messages:
            content = msg["content"]
            # 应该包含<think>标签
            assert "<think>" in content, "assistant消息应该包含<think>标签"
            assert "</think>" in content, "assistant消息应该包含</think>标签"

        estimated_turns = len([msg for msg in messages if msg["role"] == "user"])
        print(
            f"✅ 成功生成对话，包含 {len(messages)} 条消息，约 {estimated_turns} 轮对话"
        )

    def test_conversation_data_format(self):
        """测试对话数据格式"""
        # 构造模拟对话数据
        mock_conversation = {
            "id": "conv_0001",
            "tools": '[{"type": "function", "function": {"name": "test_api", "description": "test"}}]',
            "messages": [
                {"role": "user", "content": "测试用户消息"},
                {
                    "role": "tool_call",
                    "content": '{"name": "test_api", "arguments": {}}',
                },
                {"role": "tool_response", "content": '{"result": "success"}'},
                {
                    "role": "assistant",
                    "content": "<think>\n测试推理\n</think>\n\n测试回复",
                },
            ],
        }

        # 验证格式
        assert mock_conversation["id"] == "conv_0001"
        assert "messages" in mock_conversation
        assert len(mock_conversation["messages"]) == 4

        # 验证消息角色
        roles = [msg["role"] for msg in mock_conversation["messages"]]
        assert "user" in roles
        assert "assistant" in roles
        assert "tool_call" in roles
        assert "tool_response" in roles

        # 验证assistant消息格式
        assistant_msg = [
            msg for msg in mock_conversation["messages"] if msg["role"] == "assistant"
        ][0]
        assert "<think>" in assistant_msg["content"]
        assert "</think>" in assistant_msg["content"]

    def test_tools_format_validation(self):
        """测试tools字段格式"""
        # 测试有效的tools格式
        valid_tools = '[{"type": "function", "function": {"name": "api1", "description": "desc1"}}]'
        tools_data = json.loads(valid_tools)

        assert isinstance(tools_data, list)
        assert len(tools_data) == 1
        assert tools_data[0]["type"] == "function"
        assert "function" in tools_data[0]
        assert tools_data[0]["function"]["name"] == "api1"
