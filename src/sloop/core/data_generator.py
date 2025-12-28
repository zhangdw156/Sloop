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
    基于CrewAI的数据生成器
    简化为单个Agent直接生成对话数据
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

        # 创建单个对话生成Agent
        self.agent = self._create_agent()

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
        生成单个对话

        Args:
            sampled_apis: 采样得到的API子集

        Returns:
            生成的对话数据
        """
        # 构建API信息文本
        api_list_text = "\n".join([
            f"- {api['name']}: {api.get('description', 'No description')}\n  Parameters: {api.get('parameters', {})}"
            for api in sampled_apis
        ])

        # 创建对话生成任务
        task = Task(
            description=dedent(f"""
                基于以下API，生成一个多轮对话数据，包含工具调用：

                可用的API:
                {api_list_text}

                请生成一个JSON格式的对话数据，必须包含：
                {{
                  "messages": [
                    {{"role": "user", "content": "用户查询"}},
                    {{"role": "assistant", "content": "助手回复", "tool_calls": [
                      {{"id": "call_1", "type": "function", "function": {{"name": "api_name", "arguments": {{...}} }} }}
                    ]}},
                    {{"role": "tool", "content": "API执行结果", "tool_call_id": "call_1"}}
                  ],
                  "apis_used": ["{sampled_apis[0]['name']}"],
                  "user_profile": "描述用户类型"
                }}

                要求：
                1. 生成5-8轮对话
                2. 包含实际的工具调用
                3. 消息格式严格符合标准
                4. 确保对话逻辑合理
            """),
            expected_output="JSON格式的对话数据",
            agent=self.agent
        )

        # 创建临时的Crew来执行任务
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=config.verbose
        )

        # 执行任务
        result = crew.kickoff()

        # 解析结果
        return self._parse_crew_result(str(result), str(result), sampled_apis)

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
