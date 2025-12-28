"""
对话角色Agent模块
实现User/Assistant/Service三个核心对话角色
"""

from typing import List, Dict, Any, Optional
from crewai import Agent, Task, LLM
from textwrap import dedent
from .config import config
from .user_profiles import UserBehaviorSimulator


class ConversationRoleAgents:
    """
    三个核心对话角色Agent协调器
    负责生成完整的多轮对话
    """

    def __init__(self, user_profile: Dict[str, Any], apis: List[Dict[str, Any]]):
        """
        初始化对话角色Agent

        Args:
            user_profile: 用户画像
            apis: 可用的API列表
        """
        self.user_profile = user_profile
        self.apis = apis
        self.api_map = {api['name']: api for api in apis}

        # 初始化LLM
        self.llm = LLM(
            model=config.strong.model_name,
            api_key=config.strong.api_key,
            base_url=config.strong.base_url
        )

        # 创建三个核心Agent
        self.agents = self._create_agents()

        # 对话历史
        self.conversation_history = []

    def _create_agents(self) -> Dict[str, Agent]:
        """创建三个核心对话角色Agent"""

        # 1. User Agent - 基于用户画像模拟用户行为
        user_agent = Agent(
            role="用户模拟器",
            goal=f"模拟{self.user_profile['type']}类型用户的对话行为",
            backstory=dedent(f"""
                你正在模拟一个{self.user_profile['type']}类型的用户。
                用户画像：{self.user_profile}

                你的行为应该符合这个画像：
                - 沟通风格：{self.user_profile.get('communication_style', '一般')}
                - 错误处理：{self.user_profile.get('error_handling', '一般')}
                - 交互模式：{self.user_profile.get('interaction_pattern', '一般')}

                你需要根据对话上下文自然地回复用户问题。
            """),
            llm=self.llm,
            allow_delegation=False
        )

        # 2. Assistant Agent - 生成助手回复和工具调用
        assistant_agent = Agent(
            role="智能助手",
            goal="根据用户查询生成合适的回复和工具调用",
            backstory=dedent("""
                你是一个智能助手，可以调用工具来帮助用户解决问题。
                当用户提出需求时，你需要：
                1. 理解用户意图
                2. 决定是否需要调用工具
                3. 如果需要调用工具，用正确的格式生成工具调用
                4. 基于工具结果生成最终回复

                工具调用格式：
                推理过程：<tool_call>思考内容</tool_call>
                工具调用：<tool_call>{{"name": "api_name", "arguments": {{...}}}}</tool_call>
            """),
            llm=self.llm,
            allow_delegation=False
        )

        # 3. Service Agent - 模拟API调用结果
        service_agent = Agent(
            role="服务模拟器",
            goal="模拟API调用的执行结果",
            backstory=dedent(f"""
                你是API服务的结果模拟器。
                可用的API：{[api['name'] for api in self.apis]}

                当接收到工具调用请求时，你需要：
                1. 理解调用的API和参数
                2. 模拟真实的API执行结果
                3. 返回适当格式的执行结果
                4. 考虑可能的错误情况
            """),
            llm=self.llm,
            allow_delegation=False
        )

        return {
            "user": user_agent,
            "assistant": assistant_agent,
            "service": service_agent
        }

    def generate_conversation(self, initial_problem: str, target_turns: int = 10) -> Dict[str, Any]:
        """
        生成完整的多轮对话

        Args:
            initial_problem: 初始问题
            target_turns: 目标对话轮数

        Returns:
            完整的对话数据
        """
        self.conversation_history = []

        # 创建用户行为模拟器
        user_simulator = UserBehaviorSimulator(self.user_profile)

        # 生成初始用户查询
        initial_query = user_simulator.generate_initial_request(initial_problem, self.apis)
        self.conversation_history.append({
            "role": "user",
            "content": initial_query
        })

        # 计算目标轮数（允许±40%偏差）
        min_turns = max(3, int(target_turns * 0.6))
        max_turns = int(target_turns * 1.4)
        actual_turns = min(max_turns, max(min_turns, target_turns))

        # 生成多轮对话
        current_turn = 1

        while current_turn < actual_turns:
            try:
                # Assistant回复和可能的工具调用
                assistant_response = self._generate_assistant_response()

                # 检查是否包含工具调用
                if self._has_tool_call(assistant_response):
                    # 执行工具调用
                    tool_result = self._execute_tool_call(assistant_response)

                    # 基于工具结果生成最终回复
                    final_response = self._generate_final_response(tool_result)

                    self.conversation_history.append({
                        "role": "assistant",
                        "content": final_response
                    })

                    # 检查是否应该继续对话
                    if self._should_end_conversation():
                        break

                else:
                    # 直接回复，无需工具调用
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": assistant_response
                    })

                    # 检查是否应该继续对话
                    if self._should_end_conversation():
                        break

                # 生成用户后续回复（如果还没结束）
                if current_turn < actual_turns - 1:
                    user_followup = self._generate_user_followup()
                    if user_followup:
                        self.conversation_history.append({
                            "role": "user",
                            "content": user_followup
                        })
                    else:
                        break

                current_turn += 1

            except Exception as e:
                print(f"对话生成错误: {e}")
                break

        # 提取工具调用信息
        tool_calls = self._extract_tool_calls()

        return {
            "problem": initial_problem,
            "user_profile": self.user_profile,
            "conversation": self.conversation_history,
            "label": {
                "tool_call": tool_calls[-1] if tool_calls else {},
                "thought_process": self._extract_thought_process()
            },
            "apis_used": list(set(call.get("name", "") for call in tool_calls if call)),
            "total_turns": len([msg for msg in self.conversation_history if msg["role"] != "tool"])
        }

    def _generate_assistant_response(self) -> str:
        """生成助手回复"""
        # 这里应该调用Assistant Agent
        # 暂时使用简化实现
        conversation_text = self._format_conversation_history()

        prompt = dedent(f"""
            根据以下对话历史，作为智能助手回复用户：

            对话历史：
            {conversation_text}

            如果需要调用工具，请使用以下格式：
            推理：<tool_call>你的推理过程</tool_call>
            调用：<tool_call>{{"name": "api_name", "arguments": {{...}}}}</tool_call>

            回复：
        """)

        # 模拟LLM调用
        response = self.llm.call([{"role": "user", "content": prompt}])
        return response.choices[0].message.content if hasattr(response, 'choices') else "我来帮你处理这个问题。"

    def _has_tool_call(self, response: str) -> bool:
        """检查回复是否包含工具调用"""
        return "<tool_call>" in response and "{" in response

    def _execute_tool_call(self, assistant_response: str) -> Dict[str, Any]:
        """执行工具调用"""
        tool_call = self._extract_tool_call_json(assistant_response)

        if not tool_call or "name" not in tool_call:
            return {"error": "Invalid tool call"}

        api_name = tool_call["name"]
        if api_name not in self.api_map:
            return {"error": f"API {api_name} not found"}

        # 模拟API执行结果
        result = {
            "result": "success",
            "data": {
                "message": f"{api_name} 执行成功",
                "api_called": api_name,
                "parameters": tool_call.get("arguments", {})
            }
        }

        # 添加到对话历史
        self.conversation_history.append({
            "role": "tool",
            "content": result
        })

        return result

    def _generate_final_response(self, tool_result: Dict[str, Any]) -> str:
        """基于工具结果生成最终回复"""
        conversation_text = self._format_conversation_history()

        prompt = dedent(f"""
            基于工具执行结果，生成对用户的最终回复：

            对话历史：
            {conversation_text}

            工具结果：{tool_result}

            请用自然语言回复用户：
        """)

        response = self.llm.call([{"role": "user", "content": prompt}])
        return response.choices[0].message.content if hasattr(response, 'choices') else "操作已完成。"

    def _generate_user_followup(self) -> Optional[str]:
        """生成用户后续回复"""
        # 基于用户画像决定是否继续对话
        user_type = self.user_profile.get('type', 'careful')

        # 不同用户类型的跟进概率
        followup_probabilities = {
            'careful': 0.3,    # 细心用户可能会确认结果
            'careless': 0.7,   # 粗心用户可能会发现错误
            'unclear': 0.6,    # 表达不清的用户可能会继续澄清
            'curious': 0.8,    # 好奇用户喜欢继续探索
            'technical': 0.5,  # 技术用户可能会问细节
            'business': 0.2,   # 商务用户倾向于结束对话
            'novice': 0.4      # 新手用户可能需要更多指导
        }

        import random
        if random.random() < followup_probabilities.get(user_type, 0.3):
            # 生成后续问题
            conversation_text = self._format_conversation_history()

            prompt = dedent(f"""
                作为{user_type}类型的用户，根据对话历史生成一个自然的后续问题或回复：

                用户画像：{self.user_profile}
                对话历史：
                {conversation_text}

                如果不需要继续对话，请回复"结束"，否则生成一个自然的问题：
            """)

            response = self.llm.call([{"role": "user", "content": prompt}])
            followup = response.choices[0].message.content if hasattr(response, 'choices') else ""

            return None if followup.strip() == "结束" else followup.strip()

        return None

    def _should_end_conversation(self) -> bool:
        """判断是否应该结束对话"""
        # 简单的结束条件：如果最后一条助手消息解决了问题
        if not self.conversation_history:
            return False

        last_message = self.conversation_history[-1]
        if last_message["role"] == "assistant":
            content = last_message["content"].lower()
            # 检查是否包含结束语
            end_phrases = ["完成了", "解决了", "搞定了", "完成了", "done", "finished"]
            return any(phrase in content for phrase in end_phrases)

        return False

    def _format_conversation_history(self) -> str:
        """格式化对话历史"""
        formatted = []
        for msg in self.conversation_history[-5:]:  # 只保留最近5轮
            role = "用户" if msg["role"] == "user" else "助手" if msg["role"] == "assistant" else "工具"
            formatted.append(f"{role}: {msg['content']}")

        return "\n".join(formatted)

    def _extract_tool_call_json(self, response: str) -> Optional[Dict[str, Any]]:
        """从回复中提取工具调用JSON"""
        import re
        import json

        # 查找JSON格式的工具调用
        json_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        match = re.search(json_pattern, response, re.DOTALL)

        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        return None

    def _extract_tool_calls(self) -> List[Dict[str, Any]]:
        """提取对话中的所有工具调用"""
        tool_calls = []

        for msg in self.conversation_history:
            if msg["role"] == "assistant":
                tool_call = self._extract_tool_call_json(msg["content"])
                if tool_call:
                    tool_calls.append(tool_call)

        return tool_calls

    def _extract_thought_process(self) -> str:
        """提取推理过程"""
        for msg in self.conversation_history:
            if msg["role"] == "assistant":
                import re
                thought_pattern = r'<tool_call>(.*?)</tool_call>'
                match = re.search(thought_pattern, msg["content"], re.DOTALL)
                if match:
                    return match.group(1).strip()

        return ""
