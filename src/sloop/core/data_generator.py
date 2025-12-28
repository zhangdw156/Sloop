"""
基于CrewAI的数据生成模块
使用多智能体协作生成高质量的工具调用对话数据
"""

import json
from typing import List, Dict, Any, Optional
from crewai import Agent, Task, Crew, LLM
from textwrap import dedent
from .config import config
from .api_structure import APICollection


class DataGenerationCrew:
    """
    基于CrewAI的多Agent数据生成器
    使用用户/助手/服务三个角色Agent协作生成对话
    """

    def __init__(self, apis: List[Dict[str, Any]], structure_type: str = "tree"):
        """
        初始化数据生成器

        Args:
            apis: API定义列表
            structure_type: API结构化类型
        """
        self.apis = apis
        self.api_collection = APICollection(apis, structure_type)

        # 初始化LLM
        self.llm = LLM(
            model=config.strong.model_name,
            api_key=config.strong.api_key,
            base_url=config.strong.base_url
        )

        # 创建用户画像
        self.user_profiles = self._create_user_profiles()

    def _create_user_profiles(self) -> List[Dict[str, Any]]:
        """创建用户画像列表"""
        from .user_profiles import user_profile_agent

        # 使用用户画像生成器创建真实的画像
        profiles = []
        for _ in range(3):
            profile = user_profile_agent.generate_profile()
            profiles.append(profile)

        return profiles

    def _create_agent(self) -> Agent:
        """创建对话生成Agent"""
        return Agent(
            role="对话数据生成器",
            goal="基于提供的API生成高质量的多轮对话数据",
            backstory=dedent("""
                你是专业的对话数据生成专家，擅长创建包含工具调用的多轮对话。
                你能根据API规范生成真实、自然的对话场景，确保工具调用格式正确。
                你的输出必须是标准JSON格式，包含messages、apis_used等字段。
            """),
            llm=self.llm,
            allow_delegation=False
        )

    def generate_single_conversation(self, sampled_apis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        使用多Agent协作生成单个对话
        改进流程：分析采样API功能组合 → 构造综合场景 → 生成对话

        Args:
            sampled_apis: 采样得到的API子集

        Returns:
            生成的对话数据（ShareGPT格式）
        """
        try:
            from .conversation_roles import ConversationRoleAgents

            # 步骤1: 分析采样API的功能组合，构造需要使用这些API的场景
            scenario_description = self._analyze_apis_and_create_scenario(sampled_apis)
            print(f"Generated scenario: {scenario_description}")

            # 随机选择用户画像
            import random
            user_profile = random.choice(self.user_profiles)
            print(f"Selected user profile: {user_profile.get('type', 'unknown')}")

            # 步骤2: 基于场景和用户画像，生成初始用户需求
            initial_user_request = self._generate_scenario_based_request(scenario_description, user_profile)
            print(f"Generated initial request: {initial_user_request}")

            # 创建多Agent对话系统
            role_agents = ConversationRoleAgents(user_profile, sampled_apis)
            print("Created ConversationRoleAgents successfully")

            # 生成完整对话
            conversation_data = role_agents.generate_conversation(initial_user_request, target_turns=8)
            print(f"Generated conversation data type: {type(conversation_data)}")
            print(f"Conversation data keys: {list(conversation_data.keys()) if isinstance(conversation_data, dict) else 'Not a dict'}")

            # 检查conversation字段的实际内容
            conv_field = conversation_data.get('conversation', [])
            print(f"Conversation field type: {type(conv_field)}")
            if isinstance(conv_field, list):
                print(f"Conversation field length: {len(conv_field)}")
                if conv_field:
                    print(f"First conversation item: {conv_field[0]}")
            else:
                print(f"Conversation field value: {conv_field}")

            # 转换为OpenAI格式
            result = self._convert_conversation_data_to_openai_format(conversation_data)
            print(f"Final result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            return result

        except Exception as e:
            print(f"Error in generate_single_conversation: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _analyze_apis_and_create_scenario(self, sampled_apis: List[Dict[str, Any]]) -> str:
        """
        分析采样API的功能组合，构造需要使用这些API的综合场景

        Args:
            sampled_apis: 采样得到的API子集

        Returns:
            场景描述字符串
        """
        # 提取API信息
        api_info = []
        categories = set()
        for api in sampled_apis:
            name = api.get('name', '')
            description = api.get('description', '')
            category = api.get('category', 'general')
            categories.add(category)
            api_info.append(f"- {name}: {description}")

        # 基于API组合构造场景
        categories_list = list(categories)

        # 如果都是同一个类别，构造相关场景
        if len(categories) == 1:
            category = categories_list[0]
            if category == 'weather':
                scenario = "用户需要进行天气相关的综合查询和分析"
            elif category == 'travel':
                scenario = "用户需要规划一次完整的旅行行程，包括交通和餐饮安排"
            elif category == 'finance':
                scenario = "用户需要进行财务数据分析和投资决策"
            else:
                scenario = f"用户需要使用{category}相关的多个工具来完成一项综合任务"
        else:
            # 多个类别，构造跨领域场景
            if 'weather' in categories and 'travel' in categories:
                scenario = "用户需要根据天气情况规划一次户外旅行，包括查看天气预报和预订相关服务"
            elif 'finance' in categories and 'search' in categories:
                scenario = "用户需要查询市场数据并进行投资分析"
            elif 'communication' in categories and 'data' in categories:
                scenario = "用户需要处理数据并通过多种方式进行沟通和分享"
            else:
                scenario = f"用户需要综合使用{', '.join(categories_list)}多个领域的数据和服务来解决问题"

        # 添加具体API信息
        detailed_scenario = f"{scenario}。可用工具包括：\n" + "\n".join(api_info)
        return detailed_scenario

    def _generate_scenario_based_request(self, scenario_description: str, user_profile: Dict[str, Any]) -> str:
        """
        基于场景描述和用户画像生成初始用户需求

        Args:
            scenario_description: 场景描述
            user_profile: 用户画像

        Returns:
            用户的初始请求
        """
        user_type = user_profile.get('type', 'general')

        # 从场景描述中提取关键信息
        # 这里可以根据场景生成更具体的用户需求
        if '天气' in scenario_description and '旅行' in scenario_description:
            if user_type == 'careful':
                return "我想周末去北京周边旅行，但不确定天气情况，想了解一下天气预报并找找合适的餐厅。"
            elif user_type == 'curious':
                return "我对北京的天气和美食很感兴趣，想知道周末的天气怎么样，有什么推荐的餐厅吗？"
            elif user_type == 'business':
                return "我需要为下周的北京商务 trip 做准备，想了解天气情况并预订合适的餐厅。"
            else:
                return "我想去北京玩，但天气怎么样啊？有什么好吃的餐厅推荐吗？"

        elif '财务' in scenario_description or '投资' in scenario_description:
            if user_type == 'business':
                return "我想分析一下市场趋势，需要查看相关的财务数据和投资信息。"
            elif user_type == 'technical':
                return "我正在开发一个投资分析工具，需要获取市场数据和财务指标。"
            else:
                return "我想了解一下投资方面的信息，有什么数据可以看看吗？"

        else:
            # 通用场景，根据用户类型生成需求
            if user_type == 'technical':
                return "我正在开发一个应用，需要获取一些数据来测试功能。"
            elif user_type == 'business':
                return "我需要一些数据来支持我的工作决策。"
            elif user_type == 'curious':
                return "我对这个主题很感兴趣，想了解更多相关信息。"
            else:
                return "我想了解一些相关信息，能帮我查查看吗？"

    def _convert_conversation_data_to_openai_format(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        将ConversationRoleAgents的输出转换为OpenAI格式（用于SFT/RL训练）

        Args:
            conversation_data: 原始对话数据

        Returns:
            OpenAI格式的对话数据，包含tools和messages
        """
        messages = []
        apis_used = conversation_data.get('apis_used', [])

        # 获取对话历史
        conversation = conversation_data.get('conversation', [])
        i = 0

        while i < len(conversation):
            msg = conversation[i]
            role = msg.get('role')
            content = msg.get('content', '')

            if role == 'user':
                # 用户消息直接添加
                messages.append({
                    "role": "user",
                    "content": content
                })
                i += 1

            elif role == 'assistant':
                # 直接使用完整的ReAct格式内容，不移除<think>标签
                if content.strip():  # 只添加非空内容
                    messages.append({
                        "role": "assistant",
                        "content": content
                    })
                i += 1

            elif role == 'tool_call':
                # tool_call消息直接添加
                messages.append({
                    "role": "tool_call",
                    "content": content
                })
                i += 1

            elif role == 'tool_response':
                # 工具执行结果 - 已经是JSON字符串，直接使用
                messages.append({
                    "role": "tool_response",
                    "content": content
                })
                i += 1

            else:
                i += 1

        # 生成tools字符串（OpenAI格式）
        tools_json = self._generate_tools_json_for_openai(apis_used)

        return {
            "tools": tools_json,
            "messages": messages
        }

    def convert_to_qwen_format(self, openai_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        将OpenAI格式数据转换为标准SFT训练格式

        Args:
            openai_data: OpenAI格式的对话数据

        Returns:
            标准SFT训练数据格式
        """
        result = {
            "tools": openai_data["tools"],
            "messages": []
        }

        messages = openai_data["messages"]
        i = 0

        while i < len(messages):
            msg = messages[i]

            if msg["role"] == "user":
                # 用户消息保持不变
                result["messages"].append({
                    "role": "user",
                    "content": msg["content"]
                })

            elif msg["role"] == "tool_call":
                # tool_call消息已经存在，直接添加
                result["messages"].append({
                    "role": "tool_call",
                    "content": msg["content"]
                })

            elif msg["role"] == "tool_response":
                # tool_response消息已经存在，直接添加
                result["messages"].append({
                    "role": "tool_response",
                    "content": msg["content"]
                })

            elif msg["role"] == "assistant":
                # 处理助手回复，保留ReAct格式
                content = msg["content"]

                # 如果内容已经是ReAct格式，直接使用
                if content.strip():
                    result["messages"].append({
                        "role": "assistant",
                        "content": content
                    })

            i += 1

        return result

    def _build_qwen_system_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """构建Qwen格式的system prompt"""
        system_parts = [
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            "",
            "# Tools",
            "",
            "You may call one or more functions to assist with the user query.",
            "",
            "You are provided with function signatures within <tools></tools> XML tags:",
            f"<tools>{json.dumps(tools, ensure_ascii=False)}</tools>",
            "",
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:",
            "<tool_call>",
            '{"name": <function-name>, "arguments": <args-json-object>}',
            "</tool_call>"
        ]

        return "\n".join(system_parts)

    def _extract_tool_calls_from_assistant_content(self, content: str) -> List[Dict[str, Any]]:
        """从助手内容中提取工具调用"""
        # 从之前添加的工具调用消息中提取
        # 这里简化处理，因为工具调用已经在对话历史中单独添加
        return []

    def _extract_reasoning_from_assistant_content(self, content: str) -> Optional[str]:
        """从助手内容中提取推理内容"""
        if '<think>' in content and '</think>' in content:
            start = content.find('<think>')
            end = content.find('</think>') + len('</think>')
            return content[start:end]
        return None

    def _extract_final_reply_from_assistant_content(self, content: str) -> str:
        """从助手内容中提取最终回复"""
        # 移除推理和工具调用部分
        content = content.replace('<think>', '').replace('</think>', '')
        # 移除工具调用标签
        import re
        content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL)
        return content.strip()

    def _generate_tools_json_for_openai(self, apis_used: List[str]) -> str:
        """生成OpenAI格式的tools JSON字符串"""
        tools = []
        for api_name in apis_used:
            # 从self.apis中查找API定义
            api_def = next((api for api in self.apis if api['name'] == api_name), None)
            if api_def:
                tool = {
                    "type": "function",
                    "function": {
                        "name": api_def.get("name", ""),
                        "description": api_def.get("description", ""),
                        "parameters": api_def.get("parameters", {})
                    }
                }
                tools.append(tool)

        return json.dumps(tools, ensure_ascii=False)

    def _process_assistant_for_qwen(self, content: str) -> Dict[str, Any]:
        """
        处理助手内容，返回Qwen chat_template兼容的格式

        Args:
            content: 原始助手内容

        Returns:
            包含reasoning_content, tool_calls, content的字典
        """
        result = {
            "reasoning_content": None,
            "tool_calls": [],
            "content": ""
        }

        # 检查是否包含推理内容（<think>标签或Thought:）
        reasoning_content = None
        remaining_content = content

        if '<think>' in content and '</think>' in content:
            think_start = content.find('<think>')
            think_end = content.find('</think>') + len('</think>')
            reasoning_content = content[think_start:think_end]
            remaining_content = content[think_end:].strip()
        elif 'Thought:' in content:
            # 处理Thought:格式的推理
            thought_start = content.find('Thought:')
            thought_end = content.find('\n\n', thought_start)
            if thought_end == -1:
                thought_end = len(content)
            reasoning_content = f"<think>\n{content[thought_start:thought_end].replace('Thought:', '').strip()}\n</think>"
            remaining_content = content[thought_end:].strip()

        # 如果找到了推理内容，设置为reasoning_content
        if reasoning_content:
            result["reasoning_content"] = reasoning_content

        # 提取工具调用
        tool_calls = self._extract_tool_calls_from_content(remaining_content)
        if tool_calls:
            result["tool_calls"] = tool_calls
            # 移除工具调用部分，剩下的作为content
            for tool_call in tool_calls:
                remaining_content = remaining_content.replace(
                    f'<tool_call>{json.dumps(tool_call)}</tool_call>', ''
                ).strip()
            # 也移除<tool_call>标签内的解释文本
            import re
            remaining_content = re.sub(r'<tool_call>.*?</tool_call>', '', remaining_content, flags=re.DOTALL).strip()

        # 设置最终content
        result["content"] = remaining_content

        return result

    def _extract_tool_calls_from_content(self, content: str) -> List[Dict[str, Any]]:
        """从内容中提取所有工具调用"""
        import re

        tool_calls = []
        # 查找所有<tool_call>标签
        tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(tool_call_pattern, content, re.DOTALL)

        for match in matches:
            try:
                tool_call = json.loads(match.strip())
                tool_calls.append({
                    "name": tool_call.get("name", ""),
                    "arguments": tool_call.get("arguments", {})
                })
            except json.JSONDecodeError:
                continue

        return tool_calls

    def _process_assistant_content(self, content: str) -> str:
        """
        处理助手内容，确保符合ReAct格式和Qwen chat_template要求

        Args:
            content: 原始助手内容

        Returns:
            处理后的内容
        """
        # 检查是否已经包含ReAct格式
        if '<think>' in content and '</think>' in content:
            # 已经是ReAct格式，提取推理和输出部分
            think_start = content.find('<think>')
            think_end = content.find('</think>') + len('</think>')

            reasoning = content[think_start:think_end]
            remaining_content = content[think_end:].strip()

            # 检查是否有工具调用
            tool_call_match = self._extract_tool_call_pattern(remaining_content)
            if tool_call_match:
                return f"{reasoning}\n\n{tool_call_match}"
            else:
                return f"{reasoning}\n\n{remaining_content}"

        # 检查是否有工具调用标签
        if '<tool_call>' in content and '</tool_call>' in content:
            # 提取推理部分（如果有）
            import re
            tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
            tool_calls = re.findall(tool_call_pattern, content, re.DOTALL)

            if tool_calls:
                # 构建ReAct格式
                reasoning_part = content.split('<tool_call>')[0].strip()
                if reasoning_part:
                    reasoning = f"<think>\n{reasoning_part}\n</think>"
                else:
                    reasoning = "<think>\n推理过程：需要调用工具来回答用户问题\n</think>"

                # 添加工具调用
                tool_calls_text = '\n'.join(f'<tool_call>{call}</tool_call>' for call in tool_calls)

                return f"{reasoning}\n\n{tool_calls_text}"

        # 普通回复，添加默认推理
        return f"<think>\n分析用户查询并准备回复\n</think>\n\n{content}"

    def _extract_tool_call_pattern(self, content: str) -> Optional[str]:
        """从内容中提取工具调用模式"""
        import re

        # 查找<tool_call>标签
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(tool_call_pattern, content, re.DOTALL)

        if matches:
            return '\n'.join(f'<tool_call>{match.strip()}</tool_call>' for match in matches)

        return None

    def _extract_tool_call_from_content(self, content: str) -> Optional[Dict[str, Any]]:
        """从内容中提取工具调用"""
        import re

        # 查找JSON格式的工具调用
        json_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        match = re.search(json_pattern, content, re.DOTALL)

        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        return None

    def _extract_reasoning_from_content(self, content: str) -> Optional[str]:
        """从内容中提取推理部分"""
        # 移除工具调用部分，返回剩余内容作为推理
        import re
        content_without_tool_call = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL).strip()
        return content_without_tool_call if content_without_tool_call else None

    def _generate_tools_json_from_apis_used(self, apis_used: List[str]) -> str:
        """基于使用的API生成tools JSON"""
        tools = []
        for api_name in apis_used:
            # 从self.apis中查找API定义
            api_def = next((api for api in self.apis if api['name'] == api_name), None)
            if api_def:
                tool = {
                    "name": api_def.get("name", ""),
                    "description": api_def.get("description", ""),
                    "parameters": api_def.get("parameters", {})
                }
                tools.append(tool)

        return json.dumps(tools)

    def _parse_crew_result(self, conversation_output: str, quality_output: str, sampled_apis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        解析Crew执行结果，转换为ShareGPT格式

        Args:
            conversation_output: 对话生成任务的输出
            quality_output: 质量评估任务的输出
            sampled_apis: 采样API列表

        Returns:
            ShareGPT格式的字典
        """
        # 优先使用对话输出解析消息，如果失败则使用质量输出
        messages = self._extract_messages_from_output(conversation_output)
        if not messages:
            messages = self._extract_messages_from_output(quality_output)

        # 如果无法解析，返回空conversations
        if not messages:
            messages = []

        # 转换为ShareGPT格式
        conversations = self._convert_to_sharegpt_format(messages)

        # 生成tools字符串
        tools_json = self._generate_tools_json(sampled_apis)

        # 生成system提示词
        system_prompt = self._generate_system_prompt()

        return {
            "conversations": conversations,
            "tools": tools_json,
            "system": system_prompt
        }

    def _extract_messages_from_output(self, output: str) -> List[Dict[str, Any]]:
        """
        从CrewAI输出中提取消息列表

        Args:
            output: CrewAI的原始输出

        Returns:
            标准格式的消息列表
        """
        messages = []

        try:
            # 尝试从输出中查找JSON格式的消息
            import json
            import re

            # 首先尝试直接解析整个输出作为JSON
            try:
                data = json.loads(output.strip())
                if isinstance(data, dict) and "messages" in data:
                    return data["messages"]
                elif isinstance(data, dict) and "conversation" in data:
                    return data["conversation"]
            except json.JSONDecodeError:
                pass

            # 如果直接解析失败，查找可能的JSON块
            json_pattern = r'\{[^{}]*\}|\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\{[^{}]*\{[^{}]*\}[^{}]*\}[^{}]*\}'
            json_matches = re.findall(json_pattern, output, re.DOTALL)

            for match in json_matches:
                try:
                    data = json.loads(match)
                    if isinstance(data, dict) and "messages" in data:
                        return data["messages"]
                    elif isinstance(data, dict) and "conversation" in data:
                        return data["conversation"]
                    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                        # 检查是否是消息列表
                        if all(msg.get("role") and msg.get("content") for msg in data):
                            return data
                except json.JSONDecodeError:
                    continue

            # 如果没找到JSON，尝试手动解析对话模式
            # 查找用户和助手对话模式
            lines = output.split('\n')
            current_role = None
            current_content = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 检测角色切换
                if line.lower().startswith(('user:', '用户:')):
                    if current_role and current_content:
                        messages.append({
                            "role": current_role,
                            "content": '\n'.join(current_content)
                        })
                    current_role = "user"
                    current_content = [line.split(':', 1)[1].strip()]
                elif line.lower().startswith(('assistant:', '助手:', 'ai:')):
                    if current_role and current_content:
                        messages.append({
                            "role": current_role,
                            "content": '\n'.join(current_content)
                        })
                    current_role = "assistant"
                    current_content = [line.split(':', 1)[1].strip()]
                elif line.lower().startswith(('tool:', '工具:', 'system:')):
                    if current_role and current_content:
                        messages.append({
                            "role": current_role,
                            "content": '\n'.join(current_content)
                        })
                    current_role = "tool"
                    current_content = [line.split(':', 1)[1].strip()]
                elif current_role:
                    current_content.append(line)

            # 添加最后一个消息
            if current_role and current_content:
                messages.append({
                    "role": current_role,
                    "content": '\n'.join(current_content)
                })

        except Exception as e:
            print(f"Error parsing messages: {e}")

        return messages

    def _convert_to_sharegpt_format(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将OpenAI messages格式转换为ShareGPT格式

        Args:
            messages: OpenAI格式的消息列表

        Returns:
            ShareGPT格式的对话列表
        """
        conversations = []

        for msg in messages:
            # 确保msg是字典类型
            if not isinstance(msg, dict):
                print(f"Warning: msg is not dict, type={type(msg)}, value={msg}")
                continue

            role = msg.get("role")
            content = msg.get("content", "")

            if role == "user":
                conversations.append({
                    "from": "human",
                    "value": content
                })
            elif role == "assistant":
                # 检查是否有工具调用
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    # 添加助手回复（如果有的话）
                    if content.strip():
                        conversations.append({
                            "from": "gpt",
                            "value": content
                        })

                    # 添加工具调用
                    for tool_call in tool_calls:
                        function_call = {
                            "name": tool_call["function"]["name"],
                            "arguments": tool_call["function"]["arguments"]
                        }
                        conversations.append({
                            "from": "function_call",
                            "value": json.dumps(function_call)
                        })
                else:
                    # 普通助手回复
                    conversations.append({
                        "from": "gpt",
                        "value": content
                    })
            elif role == "tool":
                # 工具执行结果
                conversations.append({
                    "from": "observation",
                    "value": json.dumps(content) if isinstance(content, dict) else str(content)
                })

        return conversations

    def _generate_tools_json(self, sampled_apis: List[Dict[str, Any]]) -> str:
        """
        生成ShareGPT格式的tools JSON字符串

        Args:
            sampled_apis: 采样得到的API列表

        Returns:
            tools的JSON字符串
        """
        tools = []
        for api in sampled_apis:
            tool = {
                "name": api.get("name", ""),
                "description": api.get("description", ""),
                "parameters": api.get("parameters", {})
            }
            tools.append(tool)

        return json.dumps(tools)

    def _generate_system_prompt(self) -> str:
        """
        生成系统提示词

        Returns:
            系统提示词字符串
        """
        return dedent("""
            # Tool Calling Agent

            You are a helpful AI assistant that can use various tools to help users accomplish tasks.
            When users ask you to perform actions that require external tools or APIs, use the available tools to gather information and complete the request.

            ## Guidelines:
            - Use tools when needed to get accurate, up-to-date information
            - Explain your actions clearly to users
            - Be helpful and provide detailed responses
            - If a tool call fails, try alternative approaches
        """).strip()

    def _extract_tool_calls_from_output(self, output: str) -> Dict[str, Any]:
        """
        从输出中提取工具调用信息

        Args:
            output: CrewAI输出

        Returns:
            工具调用信息
        """
        # 从采样的API中提取可能的工具调用
        # 这里可以根据需要扩展更复杂的解析逻辑
        return {}


class BatchDataGenerator:
    """
    批量数据生成器
    支持生成大量对话数据
    """

    def __init__(self, apis: List[Dict[str, Any]], structure_type: str = "tree"):
        self.apis = apis
        self.api_collection = APICollection(apis, structure_type)
        self.generation_crew = DataGenerationCrew(apis, structure_type)

    def generate_dataset(self,
                        num_conversations: int = 100,
                        apis_per_conversation: int = 3,
                        sampling_strategy: str = "balanced",
                        target_turns: int = 10,
                        output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        生成完整的数据集

        Args:
            num_conversations: 生成对话数量
            apis_per_conversation: 每个对话使用的API数量
            sampling_strategy: API采样策略
            output_file: 输出文件路径

        Returns:
            生成的数据集
        """
        dataset = []

        for i in range(num_conversations):
            try:
                # 采样API
                sampled_apis = self.api_collection.sample_apis(
                    k=apis_per_conversation,
                    strategy=sampling_strategy
                )

                if not sampled_apis:
                    continue

                # 生成对话
                conversation_data = self.generation_crew.generate_single_conversation(sampled_apis)
                # 添加ID字段
                conversation_data["id"] = f"conv_{i+1:04d}"

                dataset.append(conversation_data)

                print(f"Generated conversation {i+1}/{num_conversations}")

            except Exception as e:
                print(f"Error generating conversation {i+1}: {e}")
                continue

        # 保存到文件
        if output_file:
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"Dataset saved to {output_file}")

        return dataset
