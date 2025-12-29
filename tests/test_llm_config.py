"""
测试 LLM 配置管理与调用封装
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from sloop.config import Settings, get_settings, reload_settings
from sloop.utils.llm import completion, chat_completion, validate_llm_config


class TestSettings:
    """测试配置管理"""

    def test_default_settings(self):
        """测试默认配置"""
        # 由于.env文件可能存在，我们只测试数据类型
        settings = Settings()
        assert isinstance(settings.model_name, str)
        assert isinstance(settings.temperature, float)
        assert isinstance(settings.max_tokens, int)
        assert isinstance(settings.timeout, int)
        assert settings.temperature >= 0.0 and settings.temperature <= 2.0
        assert settings.max_tokens > 0
        assert settings.timeout > 0

    def test_env_loading(self):
        """测试环境变量加载"""
        with patch.dict(os.environ, {
            "MODEL_NAME": "gpt-4o",
            "OPENAI_API_KEY": "test_key_123",
            "TEMPERATURE": "0.8",
            "MAX_TOKENS": "2048"
        }):
            settings = Settings()
            assert settings.model_name == "gpt-4o"
            assert settings.openai_api_key == "test_key_123"
            assert settings.temperature == 0.8
            assert settings.max_tokens == 2048

    def test_validation(self):
        """测试配置验证"""
        # 测试无API key的情况
        settings = Settings()
        settings.openai_api_key = None  # 强制设置为空
        assert not settings.validate()

        # 有API key
        settings.openai_api_key = "test_key"
        assert settings.validate()

        # 温度超出范围
        settings.temperature = 2.5
        assert not settings.validate()

        # 温度正常范围
        settings.temperature = 1.0
        settings.openai_api_key = "test_key"
        assert settings.validate()

    def test_safe_display(self):
        """测试安全显示"""
        settings = Settings()
        settings.openai_api_key = "sk-1234567890abcdef"

        display = settings.get_safe_display()
        # 应该显示前3个字符 + ***
        assert display["openai_api_key"] == "sk-1***"
        assert isinstance(display["model_name"], str)


class TestLLMUtils:
    """测试LLM工具函数"""

    @patch('sloop.utils.llm.litellm.completion')
    @patch('sloop.utils.llm.get_settings')
    def test_completion_success(self, mock_get_settings, mock_litellm):
        """测试成功调用"""
        # 模拟配置
        mock_settings = MagicMock()
        mock_settings.validate.return_value = True
        mock_settings.model_name = "gpt-4o-mini"
        mock_settings.temperature = 0.7
        mock_settings.max_tokens = 4096
        mock_settings.timeout = 60
        mock_settings.openai_api_key = "test_key"
        mock_settings.openai_api_base = None
        mock_get_settings.return_value = mock_settings

        # 模拟litellm响应
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "测试响应"
        mock_litellm.return_value = mock_response

        messages = [{"role": "user", "content": "测试"}]
        result = completion(messages)

        assert result == "测试响应"
        mock_litellm.assert_called_once()

    @patch('sloop.utils.llm.litellm.completion')
    @patch('sloop.utils.llm.get_settings')
    def test_completion_config_error(self, mock_get_settings, mock_litellm):
        """测试配置错误"""
        # 模拟无效配置
        mock_settings = MagicMock()
        mock_settings.validate.return_value = False  # 配置无效
        mock_get_settings.return_value = mock_settings

        messages = [{"role": "user", "content": "测试"}]
        result = completion(messages)

        assert result.startswith("配置错误")
        mock_litellm.assert_not_called()

    @patch('sloop.utils.llm.litellm.completion')
    @patch('sloop.utils.llm.get_settings')
    def test_completion_api_error(self, mock_get_settings, mock_litellm):
        """测试API调用错误"""
        # 模拟配置
        mock_settings = MagicMock()
        mock_settings.validate.return_value = True
        mock_settings.model_name = "gpt-4o-mini"
        mock_settings.temperature = 0.7
        mock_settings.max_tokens = 4096
        mock_settings.timeout = 60
        mock_settings.openai_api_key = "test_key"
        mock_settings.openai_api_base = None
        mock_get_settings.return_value = mock_settings

        mock_litellm.side_effect = Exception("API错误")

        messages = [{"role": "user", "content": "测试"}]
        result = completion(messages)

        assert result.startswith("调用错误")
        assert "API错误" in result

    @patch('sloop.utils.llm.litellm.completion')
    @patch('sloop.utils.llm.get_settings')
    def test_json_mode_openai(self, mock_get_settings, mock_litellm):
        """测试OpenAI JSON模式"""
        # 模拟配置
        mock_settings = MagicMock()
        mock_settings.validate.return_value = True
        mock_settings.model_name = "gpt-4"
        mock_settings.temperature = 0.7
        mock_settings.max_tokens = 4096
        mock_settings.timeout = 60
        mock_settings.openai_api_key = "test_key"
        mock_settings.openai_api_base = None
        mock_get_settings.return_value = mock_settings

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"result": "test"}'
        mock_litellm.return_value = mock_response

        messages = [{"role": "user", "content": "测试"}]
        result = completion(messages, json_mode=True)

        # 检查是否设置了response_format
        call_kwargs = mock_litellm.call_args[1]
        assert "response_format" in call_kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @patch('sloop.utils.llm.litellm.completion')
    @patch('sloop.utils.llm.get_settings')
    def test_json_mode_other_model(self, mock_get_settings, mock_litellm):
        """测试其他模型JSON模式"""
        # 模拟配置
        mock_settings = MagicMock()
        mock_settings.validate.return_value = True
        mock_settings.model_name = "claude-3"
        mock_settings.temperature = 0.7
        mock_settings.max_tokens = 4096
        mock_settings.timeout = 60
        mock_settings.openai_api_key = "test_key"
        mock_settings.openai_api_base = None
        mock_get_settings.return_value = mock_settings

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"result": "test"}'
        mock_litellm.return_value = mock_response

        messages = [{"role": "user", "content": "测试"}]
        result = completion(messages, json_mode=True)

        # 检查系统消息是否添加了JSON指令
        call_kwargs = mock_litellm.call_args[1]
        messages_arg = call_kwargs["messages"]
        assert len(messages_arg) == 2  # 原始消息 + 系统消息
        assert "请以JSON格式响应" in messages_arg[0]["content"]

    def test_chat_completion(self):
        """测试简化聊天接口"""
        with patch('sloop.utils.llm.completion') as mock_completion:
            mock_completion.return_value = "响应内容"

            result = chat_completion(
                prompt="测试提示",
                system_message="系统消息"
            )

            assert result == "响应内容"
            mock_completion.assert_called_once()

            # 检查消息格式
            call_args = mock_completion.call_args[0][0]  # 第一个参数
            assert len(call_args) == 2
            assert call_args[0]["role"] == "system"
            assert call_args[0]["content"] == "系统消息"
            assert call_args[1]["role"] == "user"
            assert call_args[1]["content"] == "测试提示"

    def test_validate_llm_config(self):
        """测试配置验证函数"""
        with patch('sloop.utils.llm.get_settings') as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.validate.return_value = True
            mock_get_settings.return_value = mock_settings

            assert validate_llm_config()

    def test_get_supported_models(self):
        """测试获取支持的模型"""
        from sloop.utils.llm import get_supported_models
        models = get_supported_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert "gpt-4o-mini" in models


def test_env_example_exists():
    """测试环境变量示例文件存在"""
    assert os.path.exists(".env.example")

    with open(".env.example", "r") as f:
        content = f.read()

    # 检查必需的配置项
    assert "OPENAI_API_KEY" in content
    assert "MODEL_NAME" in content
    assert "TEMPERATURE" in content
