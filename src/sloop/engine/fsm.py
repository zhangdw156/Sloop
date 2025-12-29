"""
有限状态机 (FSM) 核心引擎

实现对话生成的核心循环逻辑，使用 transitions 库管理状态流转。
"""

import json
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
    """FSM 状态常量 - 细粒度状态管理"""
    USER_GEN = "user_gen"
    ASSISTANT_THINK = "assistant_think"
    ASSISTANT_DECIDE = "assistant_decide"
    TOOL_CALL_GEN = "tool_call_gen"
    TOOL_EXEC = "tool_exec"
    ASSISTANT_REPLY_GEN = "assistant_reply_gen"
    EVALUATION = "evaluation"
    FINISH = "finish"


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

        # 初始化用户轮数计数器
        self.user_turn_count = 0

        # 设置状态机
        self._setup_state_machine()

        # 手动触发初始状态的回调（transitions不会自动调用）
        self.on_enter_user_gen()

        logger.info(f"🎬 ConversationLoop initialized: {self.conversation_id}")

    def _setup_state_machine(self):
        """设置状态机"""
        # 定义状态
        states = [
            FSMStates.USER_GEN,
            FSMStates.ASSISTANT_THINK,
            FSMStates.ASSISTANT_DECIDE,
            FSMStates.TOOL_CALL_GEN,
            FSMStates.TOOL_EXEC,
            FSMStates.ASSISTANT_REPLY_GEN,
            FSMStates.EVALUATION,
            FSMStates.FINISH
        ]

        # 定义状态机
        self.machine = Machine(
            model=self,
            states=states,
            initial=FSMStates.USER_GEN,
            model_attribute='current_state'
        )

        # 定义状态转换
        self.machine.add_transition('user_generated', FSMStates.USER_GEN, FSMStates.ASSISTANT_THINK)
        self.machine.add_transition('thought_generated', FSMStates.ASSISTANT_THINK, FSMStates.ASSISTANT_DECIDE)
        self.machine.add_transition('decide_tool_call', FSMStates.ASSISTANT_DECIDE, FSMStates.TOOL_CALL_GEN)
        self.machine.add_transition('decide_reply', FSMStates.ASSISTANT_DECIDE, FSMStates.ASSISTANT_REPLY_GEN)
        self.machine.add_transition('tool_calls_generated', FSMStates.TOOL_CALL_GEN, FSMStates.TOOL_EXEC)
        self.machine.add_transition('skip_tools_reply', FSMStates.TOOL_CALL_GEN, FSMStates.ASSISTANT_REPLY_GEN)  # 没有工具调用时直接回复
        self.machine.add_transition('tools_executed', FSMStates.TOOL_EXEC, FSMStates.ASSISTANT_THINK)  # ReAct 闭环
        self.machine.add_transition('reply_generated', FSMStates.ASSISTANT_REPLY_GEN, FSMStates.EVALUATION)
        self.machine.add_transition('continue_dialogue', FSMStates.EVALUATION, FSMStates.USER_GEN)
        self.machine.add_transition('finish_dialogue', FSMStates.EVALUATION, FSMStates.FINISH)
        # 允许从任何状态直接结束对话
        self.machine.add_transition('finish_dialogue', FSMStates.USER_GEN, FSMStates.FINISH)
        self.machine.add_transition('finish_dialogue', FSMStates.ASSISTANT_THINK, FSMStates.FINISH)
        self.machine.add_transition('finish_dialogue', FSMStates.ASSISTANT_DECIDE, FSMStates.FINISH)
        self.machine.add_transition('finish_dialogue', FSMStates.TOOL_CALL_GEN, FSMStates.FINISH)
        self.machine.add_transition('finish_dialogue', FSMStates.TOOL_EXEC, FSMStates.FINISH)
        self.machine.add_transition('finish_dialogue', FSMStates.ASSISTANT_REPLY_GEN, FSMStates.FINISH)

        # 注意：transitions库会自动绑定名为 on_enter_{state_name} 的方法作为状态进入回调
        # 无需手动绑定，以避免重复绑定导致的回调执行问题

    # ==================== 状态回调方法 ====================

    def on_enter_user_gen(self):
        """进入用户消息生成状态"""
        logger.info("👤 [USER_GEN] 用户消息生成")
        self.user_turn_count += 1
        print(f"👤 [USER_GEN] 用户轮次 {self.user_turn_count}")

        # 清空上一轮的缓冲区
        self.context.clear_buffers()

        # 调用用户智能体生成消息
        user_message_content = self.user_agent.generate_message(
            self.blueprint,
            self.context.messages
        )

        # 检查是否任务完成
        if self.user_agent.is_task_complete(user_message_content):
            print("   ✅ 用户表示任务完成")
            self.context.is_completed = True
            self.finish_dialogue()
            return

        # 创建消息对象并添加到上下文
        user_message = ChatMessage(role="user", content=user_message_content)
        self.context.add_message(user_message)
        print(f"   💬 用户: {user_message.content}")

        # 触发到助手思考
        self.user_generated()

    def on_enter_assistant_think(self):
        """进入助手思考状态 - 生成 CoT"""
        logger.info("🤖 [ASSISTANT_THINK] 助手正在生成思考过程")
        print(f"🤖 [ASSISTANT_THINK] 助手正在生成思考过程 (CoT)...")

        # 调用助手智能体生成思考过程
        thought_content = self.assistant_agent.generate_thought(self.context.messages)

        # 存储到上下文缓冲区
        self.context.current_thought = thought_content
        print(f"   💭 思考过程: {thought_content[:100]}...")

        # 触发到决策状态
        self.thought_generated()

    def on_enter_assistant_decide(self):
        """进入助手决策状态 - 基于思考决定下一步"""
        logger.info("🤖 [ASSISTANT_DECIDE] 助手正在决策")
        print(f"🤖 [ASSISTANT_DECIDE] 基于思考过程进行决策...")

        # 基于思考过程决定是否需要工具调用
        needs_tools = self.assistant_agent.decide_tool_use(self.context.current_thought)

        if needs_tools:
            print("   🔧 决策: 需要调用工具")
            self.decide_tool_call()
        else:
            print("   💬 决策: 直接回复")
            self.decide_reply()

    def on_enter_tool_call_gen(self):
        """进入工具调用生成状态 - 生成具体的工具调用参数"""
        logger.info("🔧 [TOOL_CALL_GEN] 生成工具调用参数")
        print(f"🔧 [TOOL_CALL_GEN] 基于思考过程生成工具调用参数...")

        # 基于思考过程生成工具调用
        tool_calls = self.assistant_agent.generate_tool_calls(self.context.current_thought, self.tools)

        if tool_calls:
            # 为每个工具调用创建独立的 tool_call 消息（扁平化格式）
            for tool_call in tool_calls:
                tool_call_data = {
                    "name": tool_call.name,
                    "arguments": tool_call.arguments
                }
                tool_call_message = ChatMessage(
                    role="tool_call",
                    content=json.dumps(tool_call_data, ensure_ascii=False)
                )
                self.context.add_message(tool_call_message)

            # 同时存储到pending列表供后续执行
            self.context.pending_tool_calls.extend(tool_calls)
            print(f"   📝 生成 {len(tool_calls)} 个工具调用消息")

            # 触发工具执行
            self.tool_calls_generated()
        else:
            print("   📝 没有生成工具调用，直接进入回复生成")
            # 如果没有工具调用，直接进入回复生成状态
            self.skip_tools_reply()

    def on_enter_tool_exec(self):
        """进入工具执行状态"""
        logger.info("🛠️ [TOOL_EXEC] 正在执行工具")
        print(f"🛠️ [TOOL_EXEC] 执行工具调用...")

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
        self.tools_executed()

    def on_enter_assistant_reply_gen(self):
        """进入助手回复生成状态 - 生成最终回复文本"""
        logger.info("🤖 [ASSISTANT_REPLY_GEN] 生成最终回复")
        print(f"🤖 [ASSISTANT_REPLY_GEN] 基于思考过程生成最终回复...")

        # 基于思考过程生成最终回复
        reply_content = self.assistant_agent.generate_reply(self.context.current_thought, self.context.messages)

        # 将思考过程和回复拼接为完整内容（用于训练数据格式）
        full_content = f"{self.context.current_thought}\n\n{reply_content}"

        # 创建助手消息（包含思考和回复）
        assistant_message = ChatMessage(
            role="assistant",
            content=full_content
        )
        self.context.add_message(assistant_message)

        print(f"   💬 助手回复: {full_content[:100]}...")

        # 触发到评估状态
        self.reply_generated()

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
            self.finish_dialogue()
            return  # 立即返回，避免后续逻辑
        else:
            print("   🔄 继续下一轮对话")
            self.continue_dialogue()

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
        while self.current_state != FSMStates.FINISH and wait_count < max_wait:
            wait_count += 1

        if self.current_state == FSMStates.FINISH:
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
