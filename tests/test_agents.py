"""
测试智能体功能
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from sloop.agents import UserAgent, AssistantAgent, ServiceAgent
from sloop.models import ToolDefinition, ChatMessage, ToolCall, Blueprint, EnvState


class TestUserAgent:
    """测试用户智能体"""

    @pytest.fixture
    def user_agent(self):
        """用户智能体fixture"""
        return UserAgent()

    @pytest.fixture
    def mock_blueprint(self):
        """模拟blueprint fixture"""
        return Blueprint(
            intent="查找餐厅并点餐",
            required_tools=["find_restaurants", "get_menu"],
            ground_truth=["find_restaurants", "get_menu"],
            initial_state={"restaurant_found": False},
            expected_state={"restaurant_found": True, "menu_loaded": True}
        )

    @pytest.fixture
    def mock_history(self):
        """模拟对话历史fixture"""
        return [
            ChatMessage(role="assistant", content="你好！有什么可以帮助你的吗？"),
            ChatMessage(role="user", content="我想找一家餐厅吃饭"),
        ]

    @patch('sloop.agents.user.chat_completion')
    def test_generate_message_success(self, mock_chat_completion, user_agent, mock_blueprint, mock_history):
        """测试成功生成用户消息"""
        mock_chat_completion.return_value = "我想找一家意大利餐厅"

        message = user_agent.generate_message(mock_blueprint, mock_history)

        assert message == "我想找一家意大利餐厅"
        assert not user_agent.is_task_complete(message)

    @patch('sloop.agents.user.chat_completion')
    def test_generate_message_stop_signal(self, mock_chat_completion, user_agent, mock_blueprint, mock_history):
        """测试生成包含停止信号的消息"""
        mock_chat_completion.return_value = "任务完成了。###STOP###"

        message = user_agent.generate_message(mock_blueprint, mock_history)

        assert message == "###STOP###"
        assert user_agent.is_task_complete(message)

    @patch('sloop.agents.user.chat_completion')
    def test_generate_message_error(self, mock_chat_completion, user_agent, mock_blueprint, mock_history):
        """测试LLM调用失败的情况"""
        mock_chat_completion.return_value = "调用错误: 网络问题"

        message = user_agent.generate_message(mock_blueprint, mock_history)

        assert "I need help with something" in message

    def test_is_task_complete(self, user_agent):
        """测试任务完成检查"""
        assert user_agent.is_task_complete("###STOP###")
        assert not user_agent.is_task_complete("继续对话")
        assert not user_agent.is_task_complete("正常消息")


class TestAssistantAgent:
    """测试助手智能体"""

    @pytest.fixture
    def mock_tools(self):
        """模拟工具fixture"""
        return [
            ToolDefinition(
                name="search_restaurants",
                description="Search for restaurants",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            )
        ]

    @pytest.fixture
    def assistant_agent(self, mock_tools):
        """助手智能体fixture"""
        return AssistantAgent(mock_tools)

    @pytest.fixture
    def mock_history(self):
        """模拟对话历史fixture"""
        return [
            ChatMessage(role="user", content="我想找餐厅"),
            ChatMessage(role="assistant", content="好的，请告诉我你在哪个城市"),
        ]

    @patch('sloop.agents.assistant.chat_completion')
    def test_generate_response_success(self, mock_chat_completion, assistant_agent, mock_history):
        """测试成功生成助手响应"""
        mock_chat_completion.return_value = "我来帮你搜索餐厅"

        response = assistant_agent.generate_response(mock_history)

        assert response == "我来帮你搜索餐厅"

    @patch('sloop.agents.assistant.chat_completion')
    def test_generate_response_error(self, mock_chat_completion, assistant_agent, mock_history):
        """测试LLM调用失败的情况"""
        mock_chat_completion.return_value = "调用错误: 网络问题"

        response = assistant_agent.generate_response(mock_history)

        assert "I encountered an error" in response

    def test_parse_tool_calls_json_format(self, assistant_agent):
        """测试解析JSON格式的工具调用"""
        response = '我来搜索餐厅{"tool_name": "search_restaurants", "arguments": {"city": "Shanghai"}}'

        tool_calls = assistant_agent.parse_tool_calls(response)

        assert len(tool_calls) == 1
        assert tool_calls[0].name == "search_restaurants"
        assert tool_calls[0].arguments == {"city": "Shanghai"}

    def test_parse_tool_calls_no_calls(self, assistant_agent):
        """测试没有工具调用的情况"""
        response = "这是一个普通的响应，没有工具调用"

        tool_calls = assistant_agent.parse_tool_calls(response)

        assert len(tool_calls) == 0
        assert not assistant_agent.should_call_tools(response)

    def test_should_call_tools(self, assistant_agent):
        """测试工具调用检测"""
        response_with_call = '调用工具{"name": "search_restaurants", "arguments": {"city": "Shanghai"}}'
        response_without_call = "普通响应"

        assert assistant_agent.should_call_tools(response_with_call)
        assert not assistant_agent.should_call_tools(response_without_call)


class TestServiceAgent:
    """测试服务智能体"""

    @pytest.fixture
    def service_agent(self):
        """服务智能体fixture"""
        return ServiceAgent()

    @pytest.fixture
    def mock_tool_call(self):
        """模拟工具调用fixture"""
        return ToolCall(
            name="search_restaurants",
            arguments={"city": "Shanghai", "cuisine": "Italian"}
        )

    @pytest.fixture
    def mock_state(self):
        """模拟状态fixture"""
        return EnvState(
            state={
                "restaurant_found": False,
                "menu_loaded": False,
                "booking_confirmed": False
            }
        )

    @pytest.fixture
    def mock_blueprint(self):
        """模拟blueprint fixture"""
        return Blueprint(
            intent="查找餐厅并预订",
            required_tools=["search_restaurants", "book_restaurant"],
            ground_truth=["search_restaurants", "book_restaurant"],
            initial_state={"restaurant_found": False},
            expected_state={"restaurant_found": True, "booking_confirmed": True}
        )

    @patch('sloop.agents.service.chat_completion')
    def test_execute_tool_success(self, mock_chat_completion, service_agent, mock_tool_call, mock_state, mock_blueprint):
        """测试成功执行工具"""
        mock_response = json.dumps({
            "response": "Found 5 Italian restaurants in Shanghai",
            "state_updates": {"restaurant_found": True, "restaurant_count": 5}
        })
        mock_chat_completion.return_value = mock_response

        result = service_agent.execute_tool(mock_tool_call, mock_state, mock_blueprint)

        assert result["response"] == "Found 5 Italian restaurants in Shanghai"
        assert result["state_updates"] == {"restaurant_found": True, "restaurant_count": 5}

    @patch('sloop.agents.service.chat_completion')
    def test_execute_tool_error(self, mock_chat_completion, service_agent, mock_tool_call, mock_state, mock_blueprint):
        """测试工具执行失败的情况"""
        mock_chat_completion.return_value = "调用错误: 网络问题"

        result = service_agent.execute_tool(mock_tool_call, mock_state, mock_blueprint)

        assert "Error executing" in result["response"]
        assert result["state_updates"] == {}

    @patch('sloop.agents.service.chat_completion')
    def test_execute_tool_invalid_json(self, mock_chat_completion, service_agent, mock_tool_call, mock_state, mock_blueprint):
        """测试返回无效JSON的情况"""
        mock_chat_completion.return_value = "这不是有效的JSON"

        result = service_agent.execute_tool(mock_tool_call, mock_state, mock_blueprint)

        assert "response parsing failed" in result["response"]
        assert result["state_updates"] == {}

    def test_update_state(self, service_agent, mock_state):
        """测试状态更新"""
        state_updates = {"restaurant_found": True, "restaurant_count": 3}

        updated_state = service_agent.update_state(mock_state, state_updates)

        assert updated_state.state["restaurant_found"] == True
        assert updated_state.state["restaurant_count"] == 3  # 新增属性
        assert updated_state.state["menu_loaded"] == False  # 未修改的属性保持不变

    def test_update_state_empty_updates(self, service_agent, mock_state):
        """测试空状态更新的情况"""
        updated_state = service_agent.update_state(mock_state, {})

        assert updated_state == mock_state  # 应该返回相同的状态


def test_render_user_prompt():
    """测试用户提示模板渲染"""
    from sloop.utils.template import render_user_prompt

    intent = "查找餐厅"
    history = [
        {"role": "assistant", "content": "你好"},
        {"role": "user", "content": "我想吃饭"}
    ]

    prompt = render_user_prompt(intent, history)

    assert intent in prompt
    assert "你好" in prompt
    assert "我想吃饭" in prompt


def test_render_assistant_prompt():
    """测试助手提示模板渲染"""
    from sloop.utils.template import render_assistant_prompt

    tools = [
        ToolDefinition(
            name="search",
            description="Search tool",
            parameters={"type": "object", "properties": {}}
        )
    ]
    history = [
        ChatMessage(role="user", content="帮我搜索")
    ]

    prompt = render_assistant_prompt(tools, history)

    assert "search" in prompt
    assert "Search tool" in prompt
    assert "帮我搜索" in prompt


def test_render_service_prompt():
    """测试服务提示模板渲染"""
    from sloop.utils.template import render_service_prompt

    tool_call = ToolCall(
        name="test_tool",
        arguments={"param": "value"}
    )

    current_state = EnvState(state={"key": "value"})
    blueprint = Blueprint(
        intent="测试意图",
        required_tools=[],
        ground_truth=[],
        initial_state={},
        expected_state={"final": True}
    )

    prompt = render_service_prompt(tool_call, current_state, blueprint)

    assert "test_tool" in prompt
    assert "测试意图" in prompt
    assert '"key": "value"' in prompt
