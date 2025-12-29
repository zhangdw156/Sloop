"""
有限状态机 (FSM) 核心引擎

实现对话生成的核心循环逻辑，使用 transitions 库管理状态流转。
"""

import random
import logging
from typing import Optional, List
from transitions import Machine

from ..models import ConversationContext, Blueprint, ChatMessage, ToolCall, ToolDefinition
from ..agents import UserAgent, AssistantAgent, ServiceAgent

# 设置日志
logger = logging.getLogger(__name__)

# 状态常量定义
class FSMStates:
    """FSM 状态常量"""
    S_INIT = "init"
    S_USER_ACTION = "user_action"
    S_ASSISTANT_THINK = "assistant_think"
    S_TOOL_EXECUTION = "tool_execution"
    S_EVALUATION = "evaluation"
    S_FINISH = "finish"


class ConversationLoop:
    """
    对话循环状态机

    管理完整的对话生成流程，从初始化到结束。
    使用 transitions.Machine 实现状态流转。
    """

    def __init__(self, blueprint: Blueprint, tools: List[ToolDefinition], conversation_id: str = None, max_turns: int = 20):
        """
        初始化对话循环

        参数:
            blueprint: 任务蓝图
            tools: 可用的工具定义列表
            conversation_id: 对话ID，如果不提供则自动生成
            max_turns: 最大对话轮数
        """
        self.blueprint = blueprint
        self.tools = tools
        self.conversation_id = conversation_id or f"conv_{random.randint(1000, 9999)}"

        # 初始化智能体
        self.user_agent = UserAgent()
        self.assistant_agent = AssistantAgent(tools)
        self.service_agent = ServiceAgent()

        # 初始化对话上下文
        self.context = ConversationContext(
            conversation_id=self.conversation_id,
            blueprint_id=getattr(blueprint, 'id', None),
            initial_state=blueprint.initial_state.copy(),
            current_user_intent=blueprint.intent,
            max_turns=max_turns
        )

        # 初始化环境状态
        self.context.env_state.update(blueprint.initial_state)

        # 设置状态机
        self._setup_state_machine()

        # 手动触发初始状态的回调（transitions不会自动调用）
        self.on_enter_init()

        logger.info(f"🎬 ConversationLoop initialized: {self.conversation_id}")

    def _setup_state_machine(self):
        """设置状态机"""
        # 定义状态
        states = [
            FSMStates.S_INIT,
            FSMStates.S_USER_ACTION,
            FSMStates.S_ASSISTANT_THINK,
            FSMStates.S_TOOL_EXECUTION,
            FSMStates.S_EVALUATION,
            FSMStates.S_FINISH
        ]

        # 定义状态机
        self.machine = Machine(
            model=self,
            states=states,
            initial=FSMStates.S_INIT,
            model_attribute='current_state'
        )

        # 定义状态转换
        self.machine.add_transition('start_conversation', FSMStates.S_INIT, FSMStates.S_USER_ACTION)
        self.machine.add_transition('user_speaks', FSMStates.S_USER_ACTION, FSMStates.S_ASSISTANT_THINK)
        self.machine.add_transition('call_tool', FSMStates.S_ASSISTANT_THINK, FSMStates.S_TOOL_EXECUTION)
        self.machine.add_transition('reply_text', FSMStates.S_ASSISTANT_THINK, FSMStates.S_EVALUATION)
        self.machine.add_transition('tool_executed', FSMStates.S_TOOL_EXECUTION, FSMStates.S_ASSISTANT_THINK)
        self.machine.add_transition('continue_conversation', FSMStates.S_EVALUATION, FSMStates.S_USER_ACTION)
        self.machine.add_transition('finish_conversation', FSMStates.S_EVALUATION, FSMStates.S_FINISH)
        # 允许从任何状态直接结束对话
        self.machine.add_transition('finish_conversation', FSMStates.S_USER_ACTION, FSMStates.S_FINISH)
        self.machine.add_transition('finish_conversation', FSMStates.S_ASSISTANT_THINK, FSMStates.S_FINISH)
        self.machine.add_transition('finish_conversation', FSMStates.S_TOOL_EXECUTION, FSMStates.S_FINISH)

        # 注意：transitions库会自动绑定名为 on_enter_{state_name} 的方法作为状态进入回调
        # 无需手动绑定，以避免重复绑定导致的回调执行问题

    # ==================== 状态回调方法 ====================

    def on_enter_init(self):
        """进入初始化状态"""
        logger.info("🔄 [INIT] 对话初始化完成")
        print(f"🔄 [INIT] 对话 {self.conversation_id} 初始化完成")
        print(f"   📋 蓝图意图: {self.blueprint.intent}")
        print(f"   🛠️ 必需工具: {self.blueprint.required_tools}")

        # 自动触发开始对话
        print("   🚀 自动开始对话...")
        self.start_conversation()
        print(f"   ✅ 状态转换完成，当前状态: {self.current_state}")

    def on_enter_user_action(self):
        """进入用户发言状态"""
        logger.info("👤 [USER_ACTION] 用户准备发言")
        print(f"👤 [USER_ACTION] 轮次 {self.context.turn_count + 1}")

        # 调用用户智能体生成消息
        user_message_content = self.user_agent.generate_message(
            self.blueprint,
            self.context.messages
        )

        # 检查是否任务完成
        if self.user_agent.is_task_complete(user_message_content):
            print("   ✅ 用户表示任务完成")
            self.context.is_completed = True
            self.finish_conversation()
            return

        # 创建消息对象并添加到上下文
        user_message = ChatMessage(role="user", content=user_message_content)
        self.context.add_message(user_message)
        print(f"   💬 用户: {user_message.content}")

        # 触发到助手思考
        self.user_speaks()

    def on_enter_assistant_think(self):
        """进入助手思考状态"""
        logger.info("🤖 [ASSISTANT_THINK] 助手正在思考")
        print(f"🤖 [ASSISTANT_THINK] 助手正在分析用户输入...")

        # 调用助手智能体生成响应
        assistant_response = self.assistant_agent.generate_response(self.context.messages)

        # 解析工具调用
        tool_calls = self.assistant_agent.parse_tool_calls(assistant_response)

        if tool_calls:
            print(f"   🔧 检测到 {len(tool_calls)} 个工具调用")
            # 将工具调用添加到pending列表
            self.context.pending_tool_calls.extend(tool_calls)

            # 创建助手消息（包含工具调用）
            assistant_message = ChatMessage(
                role="assistant",
                content=assistant_response,
                tool_call=tool_calls[0] if tool_calls else None  # 简化，假设只有一个调用
            )
            self.context.add_message(assistant_message)

            # 触发工具执行
            self.call_tool()
        else:
            print("   💬 助手直接回复")
            # 创建助手消息
            assistant_message = ChatMessage(
                role="assistant",
                content=assistant_response
            )
            self.context.add_message(assistant_message)

            # 触发回复文本
            self.reply_text()

    def on_enter_tool_execution(self):
        """进入工具执行状态"""
        logger.info("🛠️ [TOOL_EXECUTION] 正在执行工具")
        print(f"🛠️ [TOOL_EXECUTION] 执行工具调用...")

        # 处理所有pending的工具调用
        while self.context.pending_tool_calls:
            tool_call = self.context.pending_tool_calls.pop(0)

            print(f"   🔧 执行工具: {tool_call.name}")

            # 调用服务智能体执行工具
            execution_result = self.service_agent.execute_tool(
                tool_call,
                self.context.env_state,
                self.blueprint
            )

            # 更新环境状态
            if execution_result["state_updates"]:
                self.service_agent.update_state(
                    self.context.env_state,
                    execution_result["state_updates"]
                )
                print(f"   📊 状态更新: {execution_result['state_updates']}")

            # 创建工具消息
            tool_message = ChatMessage(
                role="tool",
                content=execution_result["response"],
                tool_call_id=f"call_{random.randint(1000, 9999)}"
            )
            self.context.add_message(tool_message)

            print(f"   ✅ 工具执行结果: {execution_result['response']}")

        # 返回到助手思考（ReAct闭环）
        self.tool_executed()

    def on_enter_evaluation(self):
        """进入评估状态"""
        logger.info("📊 [EVALUATION] 评估对话状态")
        print(f"📊 [EVALUATION] 评估对话状态...")

        # 如果已经完成，不要重复处理
        if self.context.is_completed:
            print("   ✅ 对话已完成，跳过评估")
            return

        self.context.increment_turn()

        # 评估结束条件（移除随机结束逻辑，确保对话充分展开）
        should_finish = (
            self.context.turn_count >= self.context.max_turns or
            self.context.env_state.validate_transition(self.blueprint.expected_state)
        )

        if should_finish:
            print("   🏁 满足结束条件，完成对话")
            self.finish_conversation()
            return  # 立即返回，避免后续逻辑
        else:
            print("   🔄 继续下一轮对话")
            self.continue_conversation()

    def on_enter_finish(self):
        """进入结束状态"""
        logger.info("✅ [FINISH] 对话完成")
        print(f"✅ [FINISH] 对话 {self.conversation_id} 完成")
        print(f"   📈 总轮次: {self.context.turn_count}")
        print(f"   📝 消息数量: {len(self.context.messages)}")
        print(f"   🎯 最终状态: {self.context.env_state.state}")



    def run(self):
        """运行完整的对话循环（同步版本，立即执行所有状态转换）"""
        logger.info("🚀 开始运行对话循环")
        print("🚀 开始运行对话循环...")

        # 在占位符实现中，所有状态转换都是同步的
        # 状态机已经在初始化时启动(on_enter_init会调用start_conversation)
        # 这里只需要等待状态机完成所有转换

        # 等待直到达到结束状态（最多等待100次，避免无限循环）
        max_wait = 100
        wait_count = 0
        while self.current_state != FSMStates.S_FINISH and wait_count < max_wait:
            wait_count += 1

        if self.current_state == FSMStates.S_FINISH:
            logger.info("🎉 对话循环运行完成")
            print("🎉 对话循环运行完成")
        else:
            logger.warning(f"⚠️ 对话循环未在{max_wait}步内完成，当前状态: {self.current_state}")
            print(f"⚠️ 对话循环未在{max_wait}步内完成，当前状态: {self.current_state}")

    # 注意：current_state 由 transitions 库自动设置，无需 property

    def get_status(self) -> dict:
        """获取当前状态信息"""
        return {
            "conversation_id": self.conversation_id,
            "current_state": self.current_state,
            "turn_count": self.context.turn_count,
            "is_completed": self.context.is_completed,
            "message_count": len(self.context.messages)
        }


# ==================== 自测代码 ====================

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 创建测试工具
    test_tools = [
        ToolDefinition(
            name="get_weather",
            description="Get weather information",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        ),
        ToolDefinition(
            name="get_location",
            description="Get user location",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]

    # 创建测试蓝图
    test_blueprint = Blueprint(
        intent="查询天气",
        required_tools=["get_weather", "get_location"],
        ground_truth=["get_weather"],
        initial_state={"weather_data": None},
        expected_state={"weather_data": "sunny"}
    )

    # 创建对话循环
    loop = ConversationLoop(test_blueprint, test_tools, "test_conv_001")

    # 运行对话
    print("=" * 50)
    print("🎬 开始FSM测试")
    print("=" * 50)

    loop.run()

    print("=" * 50)
    print("📊 最终状态:")
    print(loop.get_status())
    print("=" * 50)
