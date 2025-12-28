"""
基于CrewAI的数据生成模块
使用多智能体协作生成高质量的工具调用对话数据
"""

from typing import List, Dict, Any, Optional
from crewai import Agent, Task, Crew, LLM
from textwrap import dedent
from .config import config
from .api_structure import APICollection


class DataGenerationCrew:
    """
    基于CrewAI的数据生成团队
    包含多个专业Agent协同工作
    """

    def __init__(self, apis: List[Dict[str, Any]], structure_type: str = "tree"):
        """
        初始化数据生成Crew

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

        # 创建Agent团队
        self.agents = self._create_agents()
        self.tasks = self._create_tasks()

        # 初始化Crew
        self.crew = Crew(
            agents=list(self.agents.values()),
            tasks=self.tasks,
            verbose=config.verbose
        )

    def _create_agents(self) -> Dict[str, Agent]:
        """创建专业Agent团队"""

        # 1. API分析专家 - 分析和结构化API
        api_analyzer = Agent(
            role="API结构化专家",
            goal="分析API定义，识别功能类别和依赖关系",
            backstory=dedent("""
                你是API分析领域的专家，擅长从API定义中提取结构化信息。
                你能识别API的功能类别、参数模式和使用场景。
                你的任务是帮助系统更好地理解和组织API工具。
            """),
            llm=self.llm,
            allow_delegation=False
        )

        # 2. 场景规划师 - 基于API规划使用场景和用户画像
        scenario_planner = Agent(
            role="场景规划师",
            goal="基于可用API设计合理的使用场景、用户画像和对话规划",
            backstory=dedent("""
                你是用户体验和场景设计的专家，擅长理解用户需求。
                你能根据API功能设计出自然、实用的使用场景。
                你还需要创建合适的用户画像，定义用户的类型和行为模式。
                你的规划将为对话生成提供完整的上下文和指导。
            """),
            llm=self.llm,
            allow_delegation=True
        )

        # 3. 对话生成协调器 - 协调三个核心对话角色
        conversation_generator = Agent(
            role="对话生成协调器",
            goal="协调User/Assistant/Service三个角色生成完整的多轮对话",
            backstory=dedent("""
                你是对话生成的总协调者，负责管理整个对话流程。
                你需要根据用户画像协调三个核心角色：
                - User Agent: 基于用户画像模拟用户行为
                - Assistant Agent: 生成助手回复和工具调用
                - Service Agent: 模拟API调用结果

                你确保对话自然流畅，符合用户画像特征。
            """),
            llm=self.llm,
            allow_delegation=True
        )

        # 4. 质量评估师 - 评估生成数据的质量
        quality_assessor = Agent(
            role="质量评估师",
            goal="评估生成数据的质量和正确性",
            backstory=dedent("""
                你是数据质量和工具调用专家，擅长识别问题和改进建议。
                你能检查对话的合理性、工具调用的正确性格式规范。
                你的评估将确保数据的训练价值。
            """),
            llm=self.llm,
            allow_delegation=False
        )

        return {
            "api_analyzer": api_analyzer,
            "scenario_planner": scenario_planner,
            "conversation_generator": conversation_generator,
            "quality_assessor": quality_assessor
        }

    def _create_tasks(self) -> List[Task]:
        """创建任务链"""

        # 任务1: API分析任务
        api_analysis_task = Task(
            description=dedent(f"""
                分析提供的API列表，提取关键信息：
                1. 识别每个API的功能类别
                2. 分析API间的潜在关系和依赖
                3. 总结API的整体功能覆盖范围

                API列表: {self.apis}

                请提供结构化的分析结果。
            """),
            expected_output="结构化的API分析报告，包含类别划分和关系识别",
            agent=self.agents["api_analyzer"]
        )

        # 任务2: 场景规划任务
        scenario_planning_task = Task(
            description=dedent("""
                基于API分析结果和采样得到的API子集，设计具体的使用场景：
                1. 确定用户的主要目标和需求
                2. 设计合理的对话流程（用户查询 → 工具调用 → 结果处理）
                3. 考虑异常情况和边界条件
                4. 确保场景的实用性和多样性

                请输出详细的场景规划方案。
            """),
            expected_output="完整的场景规划方案，包含用户目标、对话流程和边界条件",
            agent=self.agents["scenario_planner"],
            context=[api_analysis_task]
        )

        # 任务3: 对话生成任务
        conversation_generation_task = Task(
            description=dedent("""
                作为对话生成协调器，你需要：
                1. 分析场景规划中的用户画像
                2. 初始化三个核心对话角色：User/Assistant/Service Agent
                3. 协调它们生成完整的多轮对话
                4. 确保对话符合用户画像特征

                对话要求：
                - User Agent: 基于用户画像模拟用户行为
                - Assistant Agent: 生成工具调用和回复
                - Service Agent: 模拟API执行结果
                - 生成目标轮数的完整对话（考虑用户画像的交互模式）

                输出完整的对话数据，包含所有角色消息和工具调用信息。
            """),
            expected_output="完整的多轮对话数据，包含用户画像、对话历史、工具调用和标签",
            agent=self.agents["conversation_generator"],
            context=[scenario_planning_task]
        )

        # 任务4: 质量评估任务
        quality_assessment_task = Task(
            description=dedent("""
                评估生成对话数据的质量：
                1. 检查对话的自然性和连贯性
                2. 验证工具调用的格式正确性
                3. 评估场景的实用性和覆盖面
                4. 识别潜在的改进点

                如果发现问题，请提供具体的改进建议。
            """),
            expected_output="质量评估报告，包含评分和改进建议",
            agent=self.agents["quality_assessor"],
            context=[conversation_generation_task]
        )

        return [
            api_analysis_task,
            scenario_planning_task,
            conversation_generation_task,
            quality_assessment_task
        ]

    def generate_single_conversation(self, sampled_apis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        生成单个对话

        Args:
            sampled_apis: 采样得到的API子集

        Returns:
            生成的对话数据
        """
        # 更新任务描述，包含采样的API
        api_list_text = "\n".join([
            f"- {api['name']}: {api.get('description', 'No description')}"
            for api in sampled_apis
        ])

        # 动态更新第一个任务的描述
        self.tasks[0].description = dedent(f"""
            分析提供的采样API列表，提取关键信息：
            1. 识别每个API的功能类别
            2. 分析API间的潜在关系和依赖
            3. 总结API的整体功能覆盖范围

            采样API列表:
            {api_list_text}

            请提供结构化的分析结果。
        """)

        # 执行Crew任务
        result = self.crew.kickoff()

        # 解析结果并格式化
        return self._parse_crew_result(result, sampled_apis)

    def _parse_crew_result(self, crew_result: Any, sampled_apis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        解析Crew执行结果，转换为标准格式

        Args:
            crew_result: Crew执行结果
            sampled_apis: 采样API列表

        Returns:
            标准格式的对话数据
        """
        # 这里需要根据Crew的输出格式解析结果
        # 由于CrewAI的输出格式可能变化，这里提供一个基础实现

        try:
            # 假设最后一个任务的输出包含最终对话
            final_output = str(crew_result)

            # 尝试解析JSON格式的对话数据
            import json
            conversation_data = json.loads(final_output)

            return {
                "problem": conversation_data.get("problem", "Generated conversation"),
                "apis_used": [api["name"] for api in sampled_apis],
                "conversation": conversation_data.get("conversation", []),
                "label": conversation_data.get("label", {}),
                "quality_score": conversation_data.get("quality_score", 0.8)
            }

        except (json.JSONDecodeError, KeyError):
            # 如果解析失败，返回基础格式
            return {
                "problem": "Generated conversation",
                "apis_used": [api["name"] for api in sampled_apis],
                "conversation": [],
                "label": {"tool_call": {}, "thought_process": ""},
                "quality_score": 0.5,
                "raw_output": str(crew_result)
            }


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
                conversation = self.generation_crew.generate_single_conversation(sampled_apis)
                conversation["id"] = f"conv_{i+1:04d}"

                dataset.append(conversation)

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
