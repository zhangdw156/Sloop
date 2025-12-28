"""
对话角色Agent模块
实现User/Assistant/Service三个核心对话角色
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from crewai import Agent, Task, LLM
from textwrap import dedent
from .config import config
from .user_profiles import UserBehaviorSimulator
from .prompt_manager import prompt_manager


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
        user_config = prompt_manager.get_agent_config(
            "user_agent",
            user_type=self.user_profile.get('type', 'general'),
            user_profile=json.dumps(self.user_profile, ensure_ascii=False),
            communication_style=self.user_profile.get('communication_style', '口语化'),
            error_handling=self.user_profile.get('error_handling', '一般'),
            interaction_pattern=self.user_profile.get('interaction_pattern', '正常交流')
        )

        user_agent = Agent(
            role=user_config["role"],
            goal=user_config["goal"],
            backstory=user_config["backstory"],
            llm=self.llm,
            allow_delegation=False
        )

        # 2. 思考Assistant Agent - 专门负责推理和工具调用决策
        thinking_config = prompt_manager.get_agent_config("thinking_agent")
        thinking_agent = Agent(
            role=thinking_config["role"],
            goal=thinking_config["goal"],
            backstory=thinking_config["backstory"],
            llm=self.llm,
            allow_delegation=False
        )

        # 3. 执行Assistant Agent - 专门负责生成最终回复
        execution_config = prompt_manager.get_agent_config("execution_agent")
        execution_agent = Agent(
            role=execution_config["role"],
            goal=execution_config["goal"],
            backstory=execution_config["backstory"],
            llm=self.llm,
            allow_delegation=False
        )

        # 4. Service Agent - 模拟API调用结果
        service_config = prompt_manager.get_agent_config(
            "service_agent",
            available_apis=[api['name'] for api in self.apis]
        )
        service_agent = Agent(
            role=service_config["role"],
            goal=service_config["goal"],
            backstory=service_config["backstory"],
            llm=self.llm,
            allow_delegation=False
        )

        return {
            "user": user_agent,
            "thinking": thinking_agent,
            "execution": execution_agent,
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
        initial_query = user_simulator.generate_initial_request(self.apis)
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
        max_iterations = actual_turns * 2  # 防止无限循环
        iteration_count = 0

        while current_turn <= actual_turns and iteration_count < max_iterations:
            try:
                iteration_count += 1

                # 步骤1: 生成思考内容
                thought_content = self._generate_assistant_response()

                # 步骤2: 决定是否需要工具调用（基于思考内容）
                needs_tool = self._decide_if_needs_tool(thought_content)

                if needs_tool:
                    # 步骤3: 生成工具调用
                    tool_call_data = self._generate_tool_call(thought_content)

                    # 添加tool_call消息到对话历史
                    self.conversation_history.append({
                        "role": "tool_call",
                        "content": json.dumps({
                            "name": tool_call_data["name"],
                            "arguments": tool_call_data["arguments"]
                        }, ensure_ascii=False)
                    })

                    # 步骤4: 执行工具调用
                    tool_result = self._execute_tool_call_from_data(tool_call_data)
                    print(f"Tool result: {tool_result}")

                    # 添加tool_response消息
                    tool_response_content = json.dumps(tool_result, ensure_ascii=False)
                    print(f"Adding tool_response: {tool_response_content}")
                    self.conversation_history.append({
                        "role": "tool_response",
                        "content": tool_response_content
                    })

                    # 步骤5: 基于工具结果生成最终回复
                    final_response = self._generate_final_response_with_agent(tool_result)

                    # 步骤6: 组合成完整的ReAct格式回复
                    combined_response = f"<think>\n{thought_content}\n</think>\n\n{final_response}"

                    # 添加完整的助手消息（包含推理+最终回复）
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": combined_response
                    })
                else:
                    # 不需要工具调用，直接生成回复
                    final_response = self._generate_direct_response(thought_content)

                    # 组合成ReAct格式
                    combined_response = f"<think>\n{thought_content}\n</think>\n\n{final_response}"

                    # 添加助手消息
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": combined_response
                    })

                # 生成用户后续回复（总是尝试生成，除非明确结束）
                user_followup = self._generate_user_followup()
                if user_followup and current_turn < actual_turns:
                    self.conversation_history.append({
                        "role": "user",
                        "content": user_followup
                    })
                    current_turn += 1
                else:
                    # 如果没有用户回复或达到轮数上限，结束对话
                    break

            except Exception as e:
                print(f"对话生成错误: {e}")
                break

        print(f"Generated {len(self.conversation_history)} messages in conversation")

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
        """生成助手回复 - 只返回推理过程，工具调用和最终回复分开处理"""
        conversation_text = self._format_conversation_history()

        # 使用Thinking Agent生成思考内容
        thinking_task_config = prompt_manager.get_task_config(
            "thinking_task",
            conversation_history=conversation_text,
            available_apis=[api['name'] for api in self.apis]
        )

        thinking_task = Task(
            description=thinking_task_config["description"],
            expected_output=thinking_task_config["expected_output"],
            agent=self.agents["thinking"]
        )

        from crewai import Crew
        thinking_crew = Crew(agents=[self.agents["thinking"]], tasks=[thinking_task], verbose=False)
        thought_content = str(thinking_crew.kickoff())

        # 返回纯思考内容（不包含<think>标签，标签由调用方添加）
        return thought_content

    def _decide_if_needs_tool(self, thought_content: str) -> bool:
        """基于思考内容决定是否需要工具调用"""
        # 简单的启发式判断：如果思考内容提到"调用"或"使用"相关API，就认为需要工具
        tool_keywords = ["调用", "使用", "查询", "获取", "需要", "应该"]
        api_names = [api['name'] for api in self.apis]

        for keyword in tool_keywords:
            if keyword in thought_content:
                # 检查是否提到了具体的API名称
                for api_name in api_names:
                    if api_name in thought_content:
                        return True

        # 默认情况下，假设大多数情况下都需要工具
        return True

    def _generate_tool_call(self, thought_content: str) -> Dict[str, Any]:
        """基于思考内容生成工具调用"""
        # 简单的实现：从思考内容中提取API名称，选择第一个匹配的API
        api_names = [api['name'] for api in self.apis]

        for api_name in api_names:
            if api_name in thought_content:
                return {
                    "name": api_name,
                    "arguments": {}  # 默认空参数
                }

        # 如果没有找到匹配的API，使用第一个可用的API
        return {
            "name": self.apis[0]['name'] if self.apis else "default_api",
            "arguments": {}
        }

    def _execute_tool_call_from_data(self, tool_call_data: Dict[str, Any]) -> Dict[str, Any]:
        """基于工具调用数据执行工具调用"""
        api_name = tool_call_data.get("name", "")
        if api_name not in self.api_map:
            return {"error": f"API {api_name} not found"}

        # 模拟API执行结果
        result = {
            "result": "success",
            "data": {
                "message": f"{api_name} 执行成功",
                "api_called": api_name,
                "parameters": tool_call_data.get("arguments", {})
            }
        }

        return result

    def _generate_direct_response(self, thought_content: str) -> str:
        """基于思考内容生成直接回复（不需要工具）"""
        # 使用execution agent生成回复，但没有工具结果
        conversation_text = self._format_conversation_history()

        execution_task_config = prompt_manager.get_task_config(
            "execution_task",
            conversation_history=conversation_text,
            thought_content=thought_content,
            tool_result="{}"  # 空工具结果
        )

        final_task = Task(
            description=execution_task_config["description"],
            expected_output=execution_task_config["expected_output"],
            agent=self.agents["execution"]
        )

        from crewai import Crew
        crew = Crew(agents=[self.agents["execution"]], tasks=[final_task], verbose=False)
        result = crew.kickoff()

        return str(result)

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

    def _generate_final_response_with_agent(self, tool_result: Dict[str, Any]) -> str:
        """基于工具结果生成最终回复 - 使用CrewAI Agent"""
        conversation_text = self._format_conversation_history()

        # 使用PromptManager获取执行任务配置
        execution_task_config = prompt_manager.get_task_config(
            "execution_task",
            conversation_history=conversation_text,
            tool_result=json.dumps(tool_result, ensure_ascii=False)
        )

        # 创建执行任务
        final_task = Task(
            description=execution_task_config["description"],
            expected_output=execution_task_config["expected_output"],
            agent=self.agents["execution"]
        )

        # 执行任务
        from crewai import Crew
        crew = Crew(agents=[self.agents["execution"]], tasks=[final_task], verbose=False)
        result = crew.kickoff()

        return str(result)

    def _generate_user_followup(self) -> Optional[str]:
        """生成用户后续回复 - 使用CrewAI Agent"""
        # 基于用户画像决定是否继续对话
        user_type = self.user_profile.get('type', 'careful')

        # 获取用户画像配置来确定跟进概率
        user_profile_config = prompt_manager.get_user_profile(user_type)
        followup_probability = 0.5  # 默认概率

        # 根据用户画像类型调整跟进概率
        if user_type == 'curious':
            followup_probability = 0.8
        elif user_type == 'careless':
            followup_probability = 0.7
        elif user_type == 'careful':
            followup_probability = 0.3
        elif user_type == 'business':
            followup_probability = 0.2
        elif user_type in ['technical', 'novice', 'unclear']:
            followup_probability = 0.5

        import random
        if random.random() < followup_probability:
            # 使用User Agent生成后续回复
            conversation_text = self._format_conversation_history()

            # 使用PromptManager获取用户任务配置
            user_task_config = prompt_manager.get_task_config(
                "user_followup_task",
                user_name=user_profile_config.get('name', '普通用户'),
                user_occupation=user_profile_config.get('occupation', '普通用户'),
                user_age=self.user_profile.get('age', '25'),
                user_personality=user_profile_config.get('personality', '普通性格'),
                user_communication_style=user_profile_config.get('communication_style', '口语化'),
                user_behavior=user_profile_config.get('typical_behavior', '正常交流'),
                conversation_history=conversation_text
            )

            user_task = Task(
                description=user_task_config["description"],
                expected_output=user_task_config["expected_output"],
                agent=self.agents["user"]
            )

            # 执行任务
            from crewai import Crew
            crew = Crew(agents=[self.agents["user"]], tasks=[user_task], verbose=False)
            result = crew.kickoff()

            followup = str(result).strip()
            return None if followup.lower().startswith("结束") else followup

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
            if msg["role"] == "tool_call":
                # tool_call消息的内容就是JSON字符串
                try:
                    tool_call = json.loads(msg["content"])
                    tool_calls.append(tool_call)
                except json.JSONDecodeError:
                    continue

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
