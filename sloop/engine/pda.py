"""
下推自动机 (PDA) 核心引擎

实现对话生成的核心循环逻辑，使用 transitions 库管理状态流转，支持栈操作。
"""

import json
import random
from typing import List

from transitions import Machine

from sloop.agents import AssistantAgent, ServiceAgent, UserAgent
from sloop.models import (
    Blueprint,
    ChatMessage,
    ConversationContext,
    ToolDefinition,
)
from sloop.utils.logger import logger

# 设置日志


# 状态常量定义
class PDAStates:
    """PDA 状态常量 - 细粒度状态管理"""

    USER_GEN = "user_gen"
    ASSISTANT_THINK = "assistant_think"
    ASSISTANT_DECIDE = "assistant_decide"
    TOOL_CALL_GEN = "tool_call_gen"
    TOOL_EXEC = "tool_exec"
    ASSISTANT_REPLY_GEN = "assistant_reply_gen"
    EVALUATION = "evaluation"
    FINISH = "finish"


class ConversationPDA:
    """
    对话循环下推自动机

    管理完整的对话生成流程，从初始化到结束。
    使用 transitions.Machine 实现状态流转，支持栈操作。
    """

    def __init__(
        self,
        blueprint: Blueprint,
        tools: List[ToolDefinition],
        conversation_id: str = None,
        max_turns: int = 20,
        auto_start: bool = True,
    ):
        """
        初始化对话循环

        参数:
            blueprint: 任务蓝图
            tools: 可用的工具定义列表
            conversation_id: 对话ID，如果不提供则自动生成
            max_turns: 最大对话轮数
            auto_start: 是否自动启动对话，默认为True。对于测试可以设为False
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
            blueprint_id=getattr(blueprint, "id", None),
            initial_state=blueprint.initial_state.copy(),
            current_user_intent=blueprint.intent,
            max_turns=max_turns,
        )

        # 初始化环境状态
        self.context.env_state.update(blueprint.initial_state)

        # 初始化用户轮数计数器
        self.user_turn_count = 0

        # 设置状态机
        self._setup_state_machine()

        # 根据auto_start参数决定是否自动启动对话
        if auto_start:
            logger.info(
                f"🎬 ConversationPDA initialized and started: {self.conversation_id}"
            )
        else:
            logger.info(
                f"🎬 ConversationPDA initialized (auto_start=False): {self.conversation_id}"
            )

    def _setup_state_machine(self):
        """设置状态机"""
        # 定义状态
        states = [
            PDAStates.USER_GEN,
            PDAStates.ASSISTANT_THINK,
            PDAStates.ASSISTANT_DECIDE,
            PDAStates.TOOL_CALL_GEN,
            PDAStates.TOOL_EXEC,
            PDAStates.ASSISTANT_REPLY_GEN,
            PDAStates.EVALUATION,
            PDAStates.FINISH,
        ]

        # 定义状态机
        self.machine = Machine(
            model=self,
            states=states,
            initial=PDAStates.USER_GEN,
            model_attribute="current_state",
        )

        # 定义状态转换
        self.machine.add_transition(
            "user_generated", PDAStates.USER_GEN, PDAStates.ASSISTANT_THINK
        )
        self.machine.add_transition(
            "thought_generated", PDAStates.ASSISTANT_THINK, PDAStates.ASSISTANT_DECIDE
        )
        self.machine.add_transition(
            "decide_tool_call", PDAStates.ASSISTANT_DECIDE, PDAStates.TOOL_CALL_GEN
        )
        self.machine.add_transition(
            "decide_reply", PDAStates.ASSISTANT_DECIDE, PDAStates.ASSISTANT_REPLY_GEN
        )
        self.machine.add_transition(
            "tool_calls_generated", PDAStates.TOOL_CALL_GEN, PDAStates.TOOL_EXEC
        )
        self.machine.add_transition(
            "skip_tools_reply", PDAStates.TOOL_CALL_GEN, PDAStates.ASSISTANT_REPLY_GEN
        )  # 没有工具调用时直接回复
        self.machine.add_transition(
            "tools_executed", PDAStates.TOOL_EXEC, PDAStates.ASSISTANT_THINK
        )  # ReAct 闭环
        self.machine.add_transition(
            "reply_generated", PDAStates.ASSISTANT_REPLY_GEN, PDAStates.EVALUATION
        )
        self.machine.add_transition(
            "continue_dialogue", PDAStates.EVALUATION, PDAStates.USER_GEN
        )
        self.machine.add_transition(
            "finish_dialogue", PDAStates.EVALUATION, PDAStates.FINISH
        )
        # 允许从任何状态直接结束对话
        self.machine.add_transition(
            "finish_dialogue", PDAStates.USER_GEN, PDAStates.FINISH
        )
        self.machine.add_transition(
            "finish_dialogue", PDAStates.ASSISTANT_THINK, PDAStates.FINISH
        )
        self.machine.add_transition(
            "finish_dialogue", PDAStates.ASSISTANT_DECIDE, PDAStates.FINISH
        )
        self.machine.add_transition(
            "finish_dialogue", PDAStates.TOOL_CALL_GEN, PDAStates.FINISH
        )
        self.machine.add_transition(
            "finish_dialogue", PDAStates.TOOL_EXEC, PDAStates.FINISH
        )
        self.machine.add_transition(
            "finish_dialogue", PDAStates.ASSISTANT_REPLY_GEN, PDAStates.FINISH
        )

        # 注意：transitions库会自动绑定名为 on_enter_{state_name} 的方法作为状态进入回调
        # 无需手动绑定，以避免重复绑定导致的回调执行问题

    def _generate_context_hint(self) -> str:
        """生成栈上下文提示信息"""
        stack_top = self.context.peek_context()
        if not stack_top or stack_top["type"] == "ROOT":
            return ""

        if stack_top["type"] == "WAITING_FOR_TOOLS":
            tool_names = stack_top["data"].get("tool_names", [])
            intent = stack_top["data"].get("intent", "未知意图")
            nested_level = stack_top["data"].get("nested_level", 0)
            indent = "  " * nested_level
            return f"{indent}系统提示：你正在等待工具结果来完成子任务。等待的工具：{', '.join(tool_names)}。任务意图：{intent}。请基于最新工具结果继续推理。"

        return ""

    def _extract_intent_from_thought(self, thought: str) -> str:
        """从思考过程中提取意图摘要"""
        if not thought:
            return "未知意图"
        # 简单提取前50个字符作为意图摘要
        return thought[:50].strip() + "..." if len(thought) > 50 else thought.strip()

    # ==================== 状态回调方法 ====================

    def _process_user_gen(self):
        """处理用户消息生成状态"""
        logger.info("👤 [USER_GEN] 用户消息生成")
        self.user_turn_count += 1
        logger.info(f"👤 [USER_GEN] 用户轮次 {self.user_turn_count}")

        # 清空上一轮的缓冲区
        self.context.clear_buffers()

        # 调用用户智能体生成消息
        user_message_content = self.user_agent.generate_message(
            self.blueprint, self.context.messages
        )

        # 检查是否任务完成，并处理停止标记
        should_stop = self.user_agent.is_task_complete(user_message_content)
        if should_stop:
            # 剥离停止标记，保留干净的消息内容
            user_message_content = user_message_content.replace(
                "###STOP###", ""
            ).strip()
            logger.info("   ✅ 用户表示任务完成")

        # 如果消息内容不为空，始终添加到对话历史
        if user_message_content:
            # 创建消息对象并添加到上下文
            user_message = ChatMessage(role="user", content=user_message_content)
            self.context.add_message(user_message)
            logger.info(f"   💬 用户: {user_message.content}")

        # 如果需要停止，则标记完成并结束对话
        if should_stop:
            self.context.is_completed = True
            return "finish_dialogue"

        # 返回下一步触发
        return "user_generated"

    def _process_assistant_think(self):
        """处理助手思考状态 - 生成 CoT"""
        logger.info("🤖 [ASSISTANT_THINK] 助手正在生成思考过程")
        logger.info("🤖 [ASSISTANT_THINK] 助手正在生成思考过程 (CoT)...")
        logger.info(
            f"   📚 当前栈状态: {[frame['type'] for frame in self.context.stack]}"
        )

        # 生成栈上下文提示
        context_hint = self._generate_context_hint()

        # 调用助手智能体生成思考过程
        thought_content = self.assistant_agent.generate_thought(
            self.context.messages, context_hint
        )

        # 存储到上下文缓冲区
        self.context.current_thought = thought_content
        logger.info(f"   💭 思考过程: {thought_content[:100]}...")

        # 返回下一步触发
        return "thought_generated"

    def _process_assistant_decide(self):
        """处理助手决策状态 - 基于思考决定下一步"""
        logger.info("🤖 [ASSISTANT_DECIDE] 助手正在决策")
        logger.info("🤖 [ASSISTANT_DECIDE] 基于思考过程进行决策...")

        # 检查栈顶是否为WAITING_FOR_TOOLS，如果是则根据决策进行POP操作
        stack_top = self.context.peek_context()
        was_waiting = stack_top and stack_top["type"] == "WAITING_FOR_TOOLS"

        # 基于思考过程决定是否需要工具调用
        needs_tools = self.assistant_agent.decide_tool_use(self.context.current_thought)

        if needs_tools:
            if was_waiting:
                # 任务进展：POP旧的WAITING帧，为新的工具调用让路
                popped = self.context.pop_context()
                logger.info(f"   📚 POP 栈: {popped['type']} - 任务进展，继续调用工具")
            logger.info("   🔧 决策: 需要调用工具")
            return "decide_tool_call"
        else:
            if was_waiting:
                # 子任务完成：POP WAITING帧
                popped = self.context.pop_context()
                logger.info(f"   📚 POP 栈: {popped['type']} - 子任务完成")
            logger.info("   💬 决策: 直接回复")
            return "decide_reply"

    def _process_tool_call_gen(self):
        """处理工具调用生成状态 - 生成具体的工具调用参数"""
        logger.info("🔧 [TOOL_CALL_GEN] 生成工具调用参数")
        logger.info("🔧 [TOOL_CALL_GEN] 基于思考过程生成工具调用参数...")

        # 基于思考过程生成工具调用
        tool_calls = self.assistant_agent.generate_tool_calls(
            self.context.current_thought, self.tools
        )

        if tool_calls:
            # PUSH 等待工具结果的上下文帧
            tool_names = [tc.name for tc in tool_calls]
            nested_level = self.context.get_stack_depth()
            self.context.push_context(
                "WAITING_FOR_TOOLS",
                {
                    "tool_names": tool_names,
                    "intent": self._extract_intent_from_thought(
                        self.context.current_thought
                    ),
                    "nested_level": nested_level,
                },
            )
            logger.info(f"   📚 PUSH 栈: WAITING_FOR_TOOLS - 工具: {tool_names}")

            # 为每个工具调用创建独立的 tool_call 消息（扁平化格式）
            for tool_call in tool_calls:
                tool_call_data = {
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                }
                tool_call_message = ChatMessage(
                    role="tool_call",
                    content=json.dumps(tool_call_data, ensure_ascii=False),
                )
                self.context.add_message(tool_call_message)

            # 同时存储到pending列表供后续执行
            self.context.pending_tool_calls.extend(tool_calls)
            logger.info(f"   📝 生成 {len(tool_calls)} 个工具调用消息")

            # 返回下一步触发
            return "tool_calls_generated"
        else:
            logger.info("   📝 没有生成工具调用，直接进入回复生成")
            # 如果没有工具调用，直接进入回复生成状态
            return "skip_tools_reply"

    def _process_tool_exec(self):
        """处理工具执行状态"""
        logger.info("🛠️ [TOOL_EXEC] 正在执行工具")
        logger.info("🛠️ [TOOL_EXEC] 执行工具调用...")

        # 处理所有pending的工具调用
        while self.context.pending_tool_calls:
            tool_call = self.context.pending_tool_calls.pop(0)

            logger.info(f"   🔧 执行工具: {tool_call.name}")

            # 调用服务智能体执行工具
            execution_result = self.service_agent.execute_tool(
                tool_call, self.context.env_state, self.blueprint
            )

            # 更新环境状态
            if execution_result["state_updates"]:
                self.service_agent.update_state(
                    self.context.env_state, execution_result["state_updates"]
                )
                logger.info(f"   📊 状态更新: {execution_result['state_updates']}")

            # 创建工具消息
            tool_message = ChatMessage(
                role="tool",
                content=execution_result["response"],
                tool_call_id=f"call_{random.randint(1000, 9999)}",
            )
            self.context.add_message(tool_message)

            logger.info(f"   ✅ 工具执行结果: {execution_result['response']}")

        # 返回到助手思考（ReAct闭环）
        return "tools_executed"

    def _process_assistant_reply_gen(self):
        """处理助手回复生成状态 - 生成最终回复文本"""
        logger.info("🤖 [ASSISTANT_REPLY_GEN] 生成最终回复")
        logger.info("🤖 [ASSISTANT_REPLY_GEN] 基于思考过程生成最终回复...")

        # 基于思考过程生成最终回复
        reply_content = self.assistant_agent.generate_reply(
            self.context.current_thought, self.context.messages
        )

        # 将思考过程和回复拼接为完整内容（用于训练数据格式）
        full_content = (
            f"<think>\n{self.context.current_thought}\n</think>\n\n{reply_content}"
        )

        # 创建助手消息（包含思考和回复）
        assistant_message = ChatMessage(role="assistant", content=full_content)
        self.context.add_message(assistant_message)

        logger.info(f"   💬 助手回复: {full_content[:100]}...")

        # 返回下一步触发
        return "reply_generated"

    def _process_evaluation(self):
        """处理评估状态"""
        logger.info("📊 [EVALUATION] 评估对话状态")
        logger.info("📊 [EVALUATION] 评估对话状态...")

        # 如果已经完成，不要重复处理
        if self.context.is_completed:
            logger.info("   ✅ 对话已完成，跳过评估")
            return "finish_dialogue"

        self.context.increment_turn()

        # 评估结束条件（移除随机结束逻辑，确保对话充分展开）
        should_finish = (
            self.context.turn_count >= self.context.max_turns
            or self.context.env_state.validate_transition(self.blueprint.expected_state)
        )

        if should_finish:
            logger.info("   🏁 满足结束条件，完成对话")
            return "finish_dialogue"
        else:
            logger.info("   🔄 继续下一轮对话")
            return "continue_dialogue"

    def on_enter_finish(self):
        """进入结束状态"""
        logger.info("✅ [FINISH] 对话完成")
        logger.info(f"✅ [FINISH] 对话 {self.conversation_id} 完成")
        logger.info(f"   📈 总轮次: {self.context.turn_count}")
        logger.info(f"   📝 消息数量: {len(self.context.messages)}")
        logger.info(f"   🎯 最终状态: {self.context.env_state.state}")

    def run(self):
        """运行完整的对话循环（循环驱动模式，避免递归溢出）"""
        logger.info("🚀 开始运行对话循环")

        while self.current_state != PDAStates.FINISH:
            trigger = None

            # 根据当前状态分发处理逻辑
            if self.current_state == PDAStates.USER_GEN:
                trigger = self._process_user_gen()
            elif self.current_state == PDAStates.ASSISTANT_THINK:
                trigger = self._process_assistant_think()
            elif self.current_state == PDAStates.ASSISTANT_DECIDE:
                trigger = self._process_assistant_decide()
            elif self.current_state == PDAStates.TOOL_CALL_GEN:
                trigger = self._process_tool_call_gen()
            elif self.current_state == PDAStates.TOOL_EXEC:
                trigger = self._process_tool_exec()
            elif self.current_state == PDAStates.ASSISTANT_REPLY_GEN:
                trigger = self._process_assistant_reply_gen()
            elif self.current_state == PDAStates.EVALUATION:
                trigger = self._process_evaluation()

            # 执行状态转换
            if trigger:
                logger.debug(f"⚡ 触发状态转换: {trigger}")
                self.trigger(trigger)

        self.on_enter_finish()

    # 注意：current_state 由 transitions 库自动设置，无需 property

    def get_status(self) -> dict:
        """获取当前状态信息"""
        return {
            "conversation_id": self.conversation_id,
            "current_state": self.current_state,
            "turn_count": self.context.turn_count,
            "is_completed": self.context.is_completed,
            "message_count": len(self.context.messages),
        }
