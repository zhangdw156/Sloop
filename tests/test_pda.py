"""
测试对话下推自动机 (ConversationPDA)

迁移自 sloop/engine/pda.py 的测试代码，并添加单元测试。
"""

# 自定义logger，用于测试日志记录
import logging
import os
from unittest.mock import MagicMock, patch

# import pytest  # 注释掉pytest，使用标准unittest
from sloop.engine.pda import ConversationPDA
from sloop.models.blueprint import Blueprint
from sloop.models.schema import ToolDefinition

# 创建logs目录（如果不存在）
test_log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(test_log_dir, exist_ok=True)

# 配置测试logger
test_logger = logging.getLogger("test_pda")
test_logger.setLevel(logging.DEBUG)

# 文件handler
log_file = os.path.join(test_log_dir, "test_pda.log")
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)

# 控制台handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)

# 添加handlers
test_logger.addHandler(file_handler)
test_logger.addHandler(console_handler)


def get_mock_tools():
    """创建模拟工具数据"""
    return [
        ToolDefinition(
            name="get_weather",
            description="Get weather information",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        ),
        ToolDefinition(
            name="get_location",
            description="Get user location",
            parameters={"type": "object", "properties": {}, "required": []},
        ),
    ]


def get_mock_blueprint():
    """创建模拟蓝图"""
    return Blueprint(
        intent="查询天气",
        required_tools=["get_weather", "get_location"],
        ground_truth=["get_weather"],
        initial_state={"weather_data": None},
        expected_state={"weather_data": "sunny"},
    )


def get_pda():
    """创建PDA实例"""
    return ConversationPDA(get_mock_blueprint(), get_mock_tools(), "test_conv_001")


def get_pda_with_mocked_agents():
    """创建PDA实例，并mock所有智能体方法"""
    # 使用auto_start=False创建PDA实例，不自动启动对话
    pda = ConversationPDA(
        get_mock_blueprint(), get_mock_tools(), "test_conv_001", auto_start=False
    )

    # Mock 用户代理
    pda.user_agent.generate_message = lambda _blueprint, _messages: "我想要查询天气"
    pda.user_agent.is_task_complete = lambda _message: False

    # Mock 助手代理 - 决策不需要工具，直接回复
    pda.assistant_agent.generate_thought = lambda _messages, _hint: (
        "我已经有了足够的信息来回答用户的问题"
    )
    pda.assistant_agent.decide_tool_use = lambda _thought: False  # 不需要工具
    pda.assistant_agent.generate_tool_calls = lambda _thought, _tools: []
    pda.assistant_agent.generate_reply = lambda _thought, _messages: (
        "根据天气信息，我来回答您的问题"
    )

    # Mock 服务代理
    pda.service_agent.execute_tool = lambda _tool_call, _env_state, _blueprint: {
        "response": "天气晴朗，温度25度",
        "state_updates": {"weather_data": "sunny"},
    }

    # Mock 状态机事件触发方法，防止状态转换
    pda.user_generated = lambda: None
    pda.thought_generated = lambda: None
    pda.decide_tool_call = lambda: None
    pda.decide_reply = lambda: None
    pda.tool_calls_generated = lambda: None
    pda.skip_tools_reply = lambda: None
    pda.tools_executed = lambda: None
    pda.reply_generated = lambda: None
    pda.continue_dialogue = lambda: None
    pda.finish_dialogue = lambda: None

    return pda


def test_initialization():
    """测试初始化"""
    test_logger.info("🔧 测试 ConversationPDA 初始化")
    pda = get_pda_with_mocked_agents()

    assert pda.blueprint == get_mock_blueprint()
    assert len(pda.tools) == 2
    assert pda.conversation_id == "test_conv_001"
    assert pda.current_state == "user_gen"
    assert pda.context is not None
    assert pda.user_turn_count == 0  # 初始化时为0

    test_logger.info("✅ 初始化测试通过")


def test_state_machine_setup():
    """测试状态机设置"""
    test_logger.info("🔄 测试状态机设置")
    pda = get_pda_with_mocked_agents()

    # 检查状态机存在
    assert hasattr(pda, "machine")
    assert pda.machine is not None

    # 检查初始状态
    assert pda.current_state == "user_gen"

    # 检查状态转换
    assert hasattr(pda, "user_generated")
    assert hasattr(pda, "thought_generated")
    assert hasattr(pda, "decide_tool_call")
    assert hasattr(pda, "decide_reply")
    assert hasattr(pda, "tool_calls_generated")
    assert hasattr(pda, "tools_executed")
    assert hasattr(pda, "reply_generated")
    assert hasattr(pda, "continue_dialogue")
    assert hasattr(pda, "finish_dialogue")

    test_logger.info("✅ 状态机设置测试通过")


def test_context_initialization():
    """测试上下文初始化"""
    test_logger.info("📚 测试上下文初始化")
    pda = get_pda_with_mocked_agents()
    mock_blueprint = get_mock_blueprint()

    assert pda.context.conversation_id == "test_conv_001"
    assert pda.context.blueprint_id == getattr(mock_blueprint, "id", None)
    assert pda.context.initial_state == mock_blueprint.initial_state
    assert pda.context.current_user_intent == mock_blueprint.intent
    assert pda.context.max_turns == 20  # 默认值
    assert pda.context.turn_count == 0
    assert not pda.context.is_completed
    assert len(pda.context.messages) == 0  # 初始化时没有消息

    test_logger.info("✅ 上下文初始化测试通过")


def test_generate_context_hint():
    """测试上下文提示生成"""
    test_logger.info("💡 测试上下文提示生成")
    pda = get_pda_with_mocked_agents()

    # 空栈
    hint = pda._generate_context_hint()
    assert hint == ""

    # WAITING_FOR_TOOLS 帧
    pda.context.push_context(
        "WAITING_FOR_TOOLS",
        {"tool_names": ["get_weather"], "intent": "查询天气", "nested_level": 0},
    )
    hint = pda._generate_context_hint()
    assert "等待工具结果" in hint
    assert "get_weather" in hint

    test_logger.info("✅ 上下文提示测试通过")


def test_extract_intent_from_thought():
    """测试意图提取"""
    test_logger.info("🎯 测试意图提取")
    pda = get_pda_with_mocked_agents()

    # 正常思考内容
    thought = "用户想要查询天气信息，需要先获取位置"
    intent = pda._extract_intent_from_thought(thought)
    assert intent == "用户想要查询天气信息，需要先获取位置"

    # 长思考内容 - 修改测试，因为53个字符的字符串截取到50个字符后会是50个字符加上"..."
    long_thought = "A" * 60  # 创建一个长字符串
    intent = pda._extract_intent_from_thought(long_thought)
    assert len(intent) <= 53  # 50 + "..."
    assert intent.endswith("...")

    # 空思考内容
    intent = pda._extract_intent_from_thought("")
    assert intent == "未知意图"

    test_logger.info("✅ 意图提取测试通过")


def test_on_enter_user_gen():
    """测试用户生成状态"""
    test_logger.info("👤 测试用户生成状态")
    pda = get_pda_with_mocked_agents()

    # 清除现有消息，以便测试
    pda.context.messages.clear()
    pda.user_turn_count = 0

    # 调用状态回调
    pda.on_enter_user_gen()

    # 检查用户轮数
    assert pda.user_turn_count == 1

    # 检查消息是否添加
    assert len(pda.context.messages) == 1
    assert pda.context.messages[0].role == "user"
    assert pda.context.messages[0].content == "我想要查询天气"

    test_logger.info("✅ 用户生成状态测试通过")


def test_on_enter_user_gen_task_complete():
    """测试任务完成的用户生成"""
    test_logger.info("✅ 测试任务完成处理")
    pda = get_pda_with_mocked_agents()

    # Mock任务完成
    pda.user_agent.is_task_complete = lambda _message: True
    pda.user_agent.generate_message = lambda _blueprint, _messages: (
        "任务完成了###STOP###"
    )

    # 清除现有消息
    pda.context.messages.clear()
    pda.user_turn_count = 0

    pda.on_enter_user_gen()

    assert pda.context.is_completed

    test_logger.info("✅ 任务完成测试通过")


def test_on_enter_assistant_think():
    """测试助手思考状态"""
    test_logger.info("🤖 测试助手思考状态")
    pda = get_pda_with_mocked_agents()

    pda.on_enter_assistant_think()

    assert pda.context.current_thought == "我已经有了足够的信息来回答用户的问题"

    test_logger.info("✅ 助手思考测试通过")


def test_on_enter_assistant_decide():
    """测试助手决策状态"""
    test_logger.info("🤖 测试助手决策状态")
    pda = get_pda_with_mocked_agents()

    # 设置当前思考
    pda.context.current_thought = "需要工具"

    # 决策需要工具
    pda.assistant_agent.decide_tool_use = lambda _thought: True
    pda.on_enter_assistant_decide()

    # 决策不需要工具
    pda.assistant_agent.decide_tool_use = lambda _thought: False
    pda.on_enter_assistant_decide()

    test_logger.info("✅ 助手决策测试通过")


def test_on_enter_tool_call_gen():
    """测试工具调用生成状态"""
    test_logger.info("🔧 测试工具调用生成")
    pda = get_pda_with_mocked_agents()

    # 设置当前思考
    pda.context.current_thought = "需要天气工具"

    # Mock生成工具调用
    mock_tool_call = MagicMock()
    mock_tool_call.name = "get_weather"
    mock_tool_call.arguments = {"location": "北京"}
    pda.assistant_agent.generate_tool_calls = lambda _thought, _tools: [mock_tool_call]

    pda.on_enter_tool_call_gen()

    # 检查工具调用是否添加到pending和消息
    assert len(pda.context.pending_tool_calls) == 1
    assert len(pda.context.messages) == 1  # 只有一条工具调用消息
    assert pda.context.messages[-1].role == "tool_call"

    # 检查栈帧是否推送
    stack_top = pda.context.peek_context()
    assert stack_top["type"] == "WAITING_FOR_TOOLS"

    test_logger.info("✅ 工具调用生成测试通过")


def test_on_enter_tool_exec():
    """测试工具执行状态"""
    test_logger.info("🛠️ 测试工具执行状态")
    pda = get_pda_with_mocked_agents()

    # 添加pending工具调用
    mock_tool_call = MagicMock()
    mock_tool_call.name = "get_weather"
    pda.context.pending_tool_calls.append(mock_tool_call)

    pda.on_enter_tool_exec()

    # 检查工具调用是否被处理
    assert len(pda.context.pending_tool_calls) == 0

    # 检查消息是否添加
    assert len(pda.context.messages) >= 1  # 至少有工具消息

    test_logger.info("✅ 工具执行测试通过")


def test_on_enter_assistant_reply_gen():
    """测试助手回复生成状态"""
    test_logger.info("🤖 测试助手回复生成")
    pda = get_pda_with_mocked_agents()

    # 设置当前思考
    pda.context.current_thought = "天气很好"

    pda.on_enter_assistant_reply_gen()

    # 检查消息是否添加
    assert len(pda.context.messages) >= 1  # 至少有助手消息
    last_message = pda.context.messages[-1]
    assert last_message.role == "assistant"
    assert "<think>" in last_message.content
    assert "根据天气信息，我来回答您的问题" in last_message.content

    test_logger.info("✅ 助手回复测试通过")


def test_on_enter_evaluation():
    """测试评估状态"""
    test_logger.info("📊 测试评估状态")
    pda = get_pda_with_mocked_agents()

    # 正常继续
    pda.context.turn_count = 0
    pda.on_enter_evaluation()
    # 应该继续对话

    # 达到最大轮数
    pda.context.turn_count = 20
    pda.on_enter_evaluation()
    # 应该结束对话

    test_logger.info("✅ 评估状态测试通过")


def test_on_enter_finish():
    """测试结束状态"""
    test_logger.info("✅ 测试结束状态")
    pda = get_pda_with_mocked_agents()

    # 注意：on_enter_finish 不会改变current_state，因为transitions库管理状态
    # 这里我们只是测试方法能正常执行
    initial_state = pda.current_state
    pda.on_enter_finish()
    # 状态应该保持不变，因为我们没有通过状态机转换

    test_logger.info("✅ 结束状态测试通过")


def test_get_status():
    """测试状态获取"""
    test_logger.info("📊 测试状态获取")
    pda = get_pda_with_mocked_agents()

    status = pda.get_status()

    assert status["conversation_id"] == "test_conv_001"
    assert status["current_state"] == "user_gen"
    assert status["turn_count"] == 0
    assert not status["is_completed"]
    assert status["message_count"] == 0  # 初始化时没有消息

    test_logger.info("✅ 状态获取测试通过")


# ==================== 集成测试（迁移自原main方法） ====================


def run_integration_test():
    """运行集成测试（原main方法逻辑）"""
    test_logger.info("🔧 ConversationPDA 集成测试")
    test_logger.info("=" * 50)

    # 创建测试工具
    test_tools = [
        ToolDefinition(
            name="get_weather",
            description="Get weather information",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        ),
        ToolDefinition(
            name="get_location",
            description="Get user location",
            parameters={"type": "object", "properties": {}, "required": []},
        ),
    ]

    # 创建测试蓝图
    test_blueprint = Blueprint(
        intent="查询天气",
        required_tools=["get_weather", "get_location"],
        ground_truth=["get_weather"],
        initial_state={"weather_data": None},
        expected_state={"weather_data": "sunny"},
    )

    # 创建对话循环
    loop = ConversationPDA(test_blueprint, test_tools, "test_conv_001")

    # 运行对话（使用mock避免实际智能体调用）
    test_logger.info("🚀 开始PDA集成测试")

    # 模拟运行几个状态转换
    test_logger.info("📊 初始状态:")
    test_logger.info(loop.get_status())

    # 手动触发一些状态转换进行测试
    with patch("sloop.agents.user.UserAgent.generate_message") as mock_user:
        mock_user.return_value = "我想要知道天气"

        loop.on_enter_user_gen()
        test_logger.info("📊 用户生成后状态:")
        test_logger.info(loop.get_status())

    with patch("sloop.agents.assistant.AssistantAgent.generate_thought") as mock_think:
        mock_think.return_value = "需要获取天气信息"

        loop.on_enter_assistant_think()
        test_logger.info("📊 思考生成后状态:")
        test_logger.info(loop.get_status())

    with patch("sloop.agents.assistant.AssistantAgent.decide_tool_use") as mock_decide:
        mock_decide.return_value = False  # 不需要工具

        loop.on_enter_assistant_decide()

    with patch("sloop.agents.assistant.AssistantAgent.generate_reply") as mock_reply:
        mock_reply.return_value = "今天天气很好"

        loop.on_enter_assistant_reply_gen()
        test_logger.info("📊 回复生成后状态:")
        test_logger.info(loop.get_status())

    loop.on_enter_evaluation()
    loop.on_enter_finish()

    test_logger.info("=" * 50)
    test_logger.info("📊 最终状态:")
    test_logger.info(loop.get_status())
    test_logger.info("=" * 50)

    test_logger.info("✅ ConversationPDA 集成测试完成！")


if __name__ == "__main__":
    # 运行集成测试
    run_integration_test()
