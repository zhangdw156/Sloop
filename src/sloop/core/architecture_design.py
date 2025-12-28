"""
Sloop项目核心架构设计
基于Coordinator-Agent-Environment的三层架构

架构说明：
1. Coordinator (协调者): 负责初始化和编排所有Agent
2. Agent层: 包含User Agent, Assistant Agent, Service Agent
3. Environment: 提供全局共享上下文和状态管理

核心原则：
- 职责分离: 每个组件有明确的单一职责
- 模块化: 高内聚，低耦合
- 可扩展性: 易于添加新的Agent类型和功能
- 标准化通信: 基于Message协议的Agent间通信
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Protocol
from dataclasses import dataclass, field
from enum import Enum
import json


# ============================================================================
# 核心数据结构
# ============================================================================

class AgentRole(Enum):
    """Agent角色枚举"""
    COORDINATOR = "coordinator"
    USER = "user"
    ASSISTANT = "assistant"
    SERVICE = "service"


class MessageType(Enum):
    """消息类型枚举"""
    USER_QUERY = "user_query"
    ASSISTANT_RESPONSE = "assistant_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SYSTEM_MESSAGE = "system_message"


@dataclass
class Message:
    """标准化消息格式"""
    type: MessageType
    role: AgentRole
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "type": self.type.value,
            "role": self.role.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


@dataclass
class ConversationContext:
    """全局对话上下文"""
    conversation_id: str
    messages: List[Message] = field(default_factory=list)
    user_profile: Dict[str, Any] = field(default_factory=dict)
    available_tools: List[Dict[str, Any]] = field(default_factory=list)
    max_turns: int = 10
    current_turn: int = 0

    def add_message(self, message: Message) -> None:
        """添加消息到上下文"""
        self.messages.append(message)

    def get_recent_messages(self, limit: int = 5) -> List[Message]:
        """获取最近的消息"""
        return self.messages[-limit:] if len(self.messages) > limit else self.messages

    def should_continue(self) -> bool:
        """判断对话是否应该继续"""
        return self.current_turn < self.max_turns


# ============================================================================
# Agent接口定义
# ============================================================================

class AgentProtocol(Protocol):
    """Agent协议接口"""

    @property
    def role(self) -> AgentRole:
        """Agent角色"""
        ...

    def process(self, context: ConversationContext) -> Message:
        """处理上下文并生成响应"""
        ...

    def can_handle(self, message: Message) -> bool:
        """判断是否能处理该消息"""
        ...


class BaseAgent(ABC):
    """Agent基类"""

    def __init__(self, role: AgentRole, config: Dict[str, Any]):
        self._role = role
        self.config = config

    @property
    def role(self) -> AgentRole:
        return self._role

    @abstractmethod
    def process(self, context: ConversationContext) -> Message:
        """处理上下文并生成响应"""
        pass

    def can_handle(self, message: Message) -> bool:
        """默认实现：根据角色判断"""
        return message.role == self._role


# ============================================================================
# Coordinator层
# ============================================================================

class Coordinator:
    """
    协调者 - 负责初始化和编排所有Agent

    职责：
    1. 从种子数据加载User Profile和Tools Definition
    2. 实例化所有Agent
    3. 控制对话流程和轮次
    4. 管理全局上下文
    5. 收集生成的对话数据
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents: Dict[AgentRole, BaseAgent] = {}
        self.context: Optional[ConversationContext] = None

    def initialize_agents(self, user_profile: Dict[str, Any],
                         tools_definition: List[Dict[str, Any]]) -> None:
        """初始化所有Agent"""
        # User Agent
        self.agents[AgentRole.USER] = UserAgent(user_profile)

        # Assistant Agent (包含Planner、Responder和Action Emitter)
        self.agents[AgentRole.ASSISTANT] = AssistantAgent({
            "planner_config": self.config.get("planner", {}),
            "responder_config": self.config.get("responder", {}),
            "action_emitter_config": self.config.get("action_emitter", {})
        })

        # Service Agent
        self.agents[AgentRole.SERVICE] = ServiceAgent(tools_definition)

    def start_conversation(self, initial_query: str,
                          max_turns: int = 10) -> ConversationContext:
        """开始新的对话"""
        conversation_id = f"conv_{len(self.agents)}"  # 简单ID生成

        self.context = ConversationContext(
            conversation_id=conversation_id,
            user_profile=self.agents[AgentRole.USER].config,
            available_tools=self.agents[AgentRole.SERVICE].config,
            max_turns=max_turns
        )

        # 添加初始用户查询
        initial_message = Message(
            type=MessageType.USER_QUERY,
            role=AgentRole.USER,
            content=initial_query
        )
        self.context.add_message(initial_message)

        return self.context

    def run_simulation(self, context: ConversationContext) -> List[Message]:
        """运行模拟对话"""
        while context.should_continue():
            # 1. Assistant处理用户查询
            assistant_response = self.agents[AgentRole.ASSISTANT].process(context)
            context.add_message(assistant_response)

            # 检查是否有工具调用
            if self._has_tool_call(assistant_response):
                # 解析工具调用（可能有多个）
                tool_calls = self._parse_tool_calls(assistant_response)

                # 2. Service执行所有工具调用
                tool_results = []
                for tool_call in tool_calls:
                    # 创建工具调用消息
                    tool_call_msg = Message(
                        type=MessageType.TOOL_CALL,
                        role=AgentRole.ASSISTANT,
                        content=json.dumps(tool_call, ensure_ascii=False)
                    )
                    context.add_message(tool_call_msg)

                    # 执行工具调用
                    tool_result = self.agents[AgentRole.SERVICE].process_tool_call(context, tool_call)
                    tool_result_msg = Message(
                        type=MessageType.TOOL_RESULT,
                        role=AgentRole.SERVICE,
                        content=json.dumps(tool_result, ensure_ascii=False)
                    )
                    context.add_message(tool_result_msg)
                    tool_results.append(tool_result)

                # 3. Assistant基于所有工具结果生成最终回复
                final_response = self.agents[AgentRole.ASSISTANT].generate_final_response(context, tool_results)
                context.add_message(final_response)

            # 4. User生成后续回复（如果需要）
            if context.current_turn < context.max_turns - 1:
                user_followup = self.agents[AgentRole.USER].process(context)
                if user_followup:
                    context.add_message(user_followup)
                    context.current_turn += 1
                else:
                    break
            else:
                break

        return context.messages

    def _parse_tool_calls(self, assistant_response: Message) -> List[Dict[str, Any]]:
        """解析助手响应中的工具调用（支持多个）"""
        try:
            response_data = json.loads(assistant_response.content)
            tool_calls_data = response_data.get("tool_calls", [])

            tool_calls = []
            for tool_call_data in tool_calls_data:
                if isinstance(tool_call_data, dict) and "name" in tool_call_data:
                    tool_calls.append(tool_call_data)

            return tool_calls
        except:
            # 如果解析失败，尝试从内容中提取单个工具调用
            return []

    def _has_tool_call(self, message: Message) -> bool:
        """检查消息是否包含工具调用"""
        try:
            response_data = json.loads(message.content)
            tool_calls = response_data.get("tool_calls", [])
            return len(tool_calls) > 0
        except:
            return False


# ============================================================================
# Agent实现
# ============================================================================

class UserAgent(BaseAgent):
    """用户Agent - 模拟真实用户行为"""

    def __init__(self, user_profile: Dict[str, Any]):
        super().__init__(AgentRole.USER, user_profile)

    def process(self, context: ConversationContext) -> Message:
        """生成用户查询或回复"""
        # 基于用户画像和对话历史生成回复
        # 这里应该调用实际的LLM来生成用户行为
        return Message(
            type=MessageType.USER_QUERY,
            role=self.role,
            content="用户生成的回复内容"
        )


class AssistantAgent(BaseAgent):
    """
    助手Agent - 包含Planner、Responder和Action Emitter三个子模块

    架构：
    - Planner: 生成Chain of Thought (推理过程)
    - Responder: 基于计划生成最终用户回复
    - Action Emitter: 负责生成工具调用
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentRole.ASSISTANT, config)
        self.planner = Planner(config.get("planner_config", {}))
        self.responder = Responder(config.get("responder_config", {}))
        self.action_emitter = ActionEmitter(config.get("action_emitter_config", {}))

    def process(self, context: ConversationContext) -> Message:
        """处理用户查询，生成回复和可能的工具调用"""

        # 1. Planner进行推理
        thought_process = self.planner.think(context)

        # 2. 检查是否需要工具调用
        if self.planner.needs_tool_call(thought_process):
            # 3. Action Emitter生成工具调用（可能多个）
            tool_calls = self.action_emitter.emit_tool_calls(context, thought_process)

            return Message(
                type=MessageType.ASSISTANT_RESPONSE,
                role=self.role,
                content=json.dumps({
                    "thought": thought_process,
                    "tool_calls": tool_calls
                }, ensure_ascii=False)
            )
        else:
            # 4. Responder生成最终回复
            final_response = self.responder.respond(context, thought_process)

            return Message(
                type=MessageType.ASSISTANT_RESPONSE,
                role=self.role,
                content=self.generate_reply(thought_process, final_response)
            )

    def generate_reply(self, reasoning: str, response: str) -> str:
        """
        生成助手对用户的回复，强制使用<think>标签格式

        Args:
            reasoning: Planner生成的推理过程
            response: Responder生成的最终回复

        Returns:
            格式化的回复字符串
        """
        # 去除推理过程中的前缀（如"Detailed reasoning: "）
        clean_reasoning = reasoning.strip()
        if clean_reasoning.startswith("Detailed reasoning:"):
            clean_reasoning = clean_reasoning[len("Detailed reasoning:"):].strip()

        # 格式化输出
        formatted_content = f"<think>\n{clean_reasoning}\n</think>\n\n{response}"

        return formatted_content

    def generate_final_response(self, context: ConversationContext, tool_results: List[Dict[str, Any]]) -> Message:
        """基于工具结果生成最终回复"""
        # Planner生成推理（基于工具结果）
        thought_process = self.planner.think(context)

        # Responder基于工具结果生成回复
        final_response = self.responder.respond_from_tools(context, tool_results)

        return Message(
            type=MessageType.ASSISTANT_RESPONSE,
            role=self.role,
            content=self.generate_reply(thought_process, final_response)
        )


class Planner:
    """规划器 - 生成Chain of Thought (推理过程)"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def think(self, context: ConversationContext) -> str:
        """生成推理过程"""
        # 检查用户查询是否需要工具调用
        user_messages = [msg for msg in context.messages if msg.role.value == "user"]
        if user_messages:
            last_user_msg = user_messages[-1].content
            # 如果用户提到天气或空气质量，说明需要调用工具
            if "天气" in last_user_msg or "空气质量" in last_user_msg or "aqi" in last_user_msg.lower():
                return "用户询问天气或空气质量信息，我需要调用相关的API工具来获取准确数据。"

        # 默认推理
        return "用户的问题可能需要调用工具来获取准确信息。"

    def needs_tool_call(self, thought: str) -> bool:
        """判断是否需要工具调用"""
        # 基于推理内容判断
        return "需要调用" in thought or "调用相关的API" in thought


class Responder:
    """响应器 - 基于计划生成最终用户回复"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def respond(self, context: ConversationContext, thought: str) -> str:
        """生成最终回复"""
        # 这里应该调用LLM生成自然语言回复
        return "基于推理生成的回复内容"

    def respond_from_tools(self, context: ConversationContext, tool_results: List[Dict[str, Any]]) -> str:
        """基于工具结果生成最终回复"""
        # 解析工具结果并生成用户友好的回复
        beijing_aqi = None
        shanghai_aqi = None

        for result in tool_results:
            city = result.get("city")
            aqi = result.get("aqi")

            if city == "北京":
                beijing_aqi = aqi
            elif city == "上海":
                shanghai_aqi = aqi

        # 生成回复
        response_parts = []
        if beijing_aqi:
            response_parts.append(f"北京今天的空气质量指数为{beijing_aqi}，属于良好水平")
        if shanghai_aqi:
            response_parts.append(f"上海今天的空气质量指数为{shanghai_aqi}，属于轻度污染水平")

        if response_parts:
            return f"根据天气预报工具，{'；'.join(response_parts)}。"
        else:
            return "根据天气预报工具，暂时无法获取空气质量数据。"


class ActionEmitter:
    """动作发射器 - 负责生成工具调用"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def emit_tool_calls(self, context: ConversationContext, thought: str) -> List[Dict[str, Any]]:
        """生成工具调用（可能多个）"""
        # 检查用户查询是否包含多个城市
        user_messages = [msg for msg in context.messages if msg.role.value == "user"]
        if user_messages:
            last_user_msg = user_messages[-1].content

            # 解析城市信息
            cities = []
            if "北京" in last_user_msg:
                cities.append("北京")
            if "上海" in last_user_msg:
                cities.append("上海")

            # 为每个城市生成工具调用
            tool_calls = []
            for city in cities:
                tool_calls.append({
                    "name": "realtime_aqi",
                    "arguments": {"city": city}
                })

            return tool_calls

        # 默认实现：为每个可用工具生成调用
        available_tools = context.available_tools
        tool_calls = []
        for tool in available_tools[:2]:  # 限制为最多2个工具调用
            tool_calls.append({
                "name": tool.get("name", "default_tool"),
                "arguments": {}  # 这里应该基于具体需求生成参数
            })

        return tool_calls


class ServiceAgent(BaseAgent):
    """服务Agent - 模拟API服务端"""

    def __init__(self, tools_definition: List[Dict[str, Any]]):
        super().__init__(AgentRole.SERVICE, tools_definition)

    def process(self, context: ConversationContext) -> Message:
        """执行工具调用并返回结果"""
        # 获取最新的工具调用消息
        recent_messages = context.get_recent_messages()
        tool_call_msg = None

        for msg in reversed(recent_messages):
            if msg.type == MessageType.TOOL_CALL:
                tool_call_msg = msg
                break

        if not tool_call_msg:
            return Message(
                type=MessageType.TOOL_RESULT,
                role=self.role,
                content=json.dumps({"error": "No tool call found"})
            )

        # 解析工具调用
        try:
            tool_call_data = json.loads(tool_call_msg.content)
            tool_call = tool_call_data.get("tool_call", {})

            # 模拟工具执行
            result = self._execute_tool(tool_call)

            return Message(
                type=MessageType.TOOL_RESULT,
                role=self.role,
                content=json.dumps(result, ensure_ascii=False)
            )

        except Exception as e:
            return Message(
                type=MessageType.TOOL_RESULT,
                role=self.role,
                content=json.dumps({"error": str(e)})
            )

    def process_tool_call(self, context: ConversationContext, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个工具调用"""
        return self._execute_tool(tool_call)

    def _execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """执行具体的工具调用"""
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})

        # 模拟不同城市的空气质量数据
        city = arguments.get("city", "")
        if city == "北京":
            return {
                "city": "北京",
                "aqi": "10",
                "unit": "celsius"
            }
        elif city == "上海":
            return {
                "city": "上海",
                "aqi": "72",
                "unit": "fahrenheit"
            }
        else:
            # 默认模拟结果
            return {
                "result": "success",
                "data": {
                    "message": f"{tool_name} 执行成功",
                    "api_called": tool_name,
                    "parameters": arguments
                }
            }


# ============================================================================
# 数据转换器
# ============================================================================

class DataConverter:
    """数据转换器 - 将Message格式转换为训练数据格式"""

    @staticmethod
    def messages_to_sft_format(messages: List[Message],
                              tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """转换为SFT训练格式"""
        # 过滤和转换消息
        sft_messages = []

        for msg in messages:
            if msg.type == MessageType.USER_QUERY:
                sft_messages.append({
                    "role": "user",
                    "content": msg.content
                })
            elif msg.type == MessageType.TOOL_CALL:
                sft_messages.append({
                    "role": "tool_call",
                    "content": msg.content
                })
            elif msg.type == MessageType.TOOL_RESULT:
                sft_messages.append({
                    "role": "tool_response",
                    "content": msg.content
                })
            elif msg.type == MessageType.ASSISTANT_RESPONSE and not msg.content.startswith("{"):
                # 只添加最终的assistant回复，跳过包含tool_calls的中间消息
                sft_messages.append({
                    "role": "assistant",
                    "content": msg.content
                })

        return {
            "tools": json.dumps(tools, ensure_ascii=False),
            "messages": sft_messages
        }


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """使用示例"""

    # 配置
    config = {
        "planner": {"model": "qwen", "temperature": 0.7},
        "action_emitter": {"strict_json": True},
        "max_turns": 10
    }

    # 创建Coordinator
    coordinator = Coordinator(config)

    # 初始化Agents
    user_profile = {"type": "technical", "communication_style": "precise"}
    tools_definition = [
        {"name": "search_api", "description": "搜索API", "parameters": {}}
    ]

    coordinator.initialize_agents(user_profile, tools_definition)

    # 开始对话
    context = coordinator.start_conversation("帮我查询天气", max_turns=5)

    # 运行模拟
    messages = coordinator.run_simulation(context)

    # 转换为训练数据
    sft_data = DataConverter.messages_to_sft_format(messages, tools_definition)

    return sft_data


def test_weather_scenario():
    """测试天气查询场景，生成用户期望的格式"""

    # 配置
    config = {
        "planner": {"model": "qwen", "temperature": 0.7},
        "action_emitter": {"strict_json": True},
        "max_turns": 10
    }

    # 创建Coordinator
    coordinator = Coordinator(config)

    # 初始化Agents - 使用天气相关的工具
    user_profile = {"type": "casual", "communication_style": "friendly"}
    tools_definition = [
        {
            "name": "realtime_aqi",
            "description": "天气预报。获取实时空气质量。当前空气质量，PM2.5，PM10信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名，例如：上海"
                    }
                },
                "required": ["city"]
            }
        }
    ]

    coordinator.initialize_agents(user_profile, tools_definition)

    # 开始对话
    context = coordinator.start_conversation("北京和上海今天的天气情况", max_turns=1)

    # 运行模拟
    messages = coordinator.run_simulation(context)

    # 转换为训练数据格式
    sft_data = DataConverter.messages_to_sft_format(messages, tools_definition)

    return sft_data


if __name__ == "__main__":
    # 运行天气场景测试
    result = test_weather_scenario()
    print(json.dumps(result, ensure_ascii=False, indent=2))
