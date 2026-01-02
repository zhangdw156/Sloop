"""
用户画像管理模块

定义和管理不同的用户画像，用于生成多样化的用户行为。
"""

from typing import List

from sloop.models.schema import UserPersona


class PersonaManager:
    """
    用户画像管理器

    提供预定义的画像集合，并根据工具链复杂度自动选择合适的画像。
    """

    def __init__(self):
        """初始化画像管理器"""
        self.personas = self._load_default_personas()

    def _load_default_personas(self) -> List[UserPersona]:
        """
        加载默认的用户画像

        返回:
            默认画像列表
        """
        return [
            UserPersona(
                name="novice_user",
                description="技术小白用户，对API和参数不熟悉",
                behavior_traits=[
                    "经常遗漏重要参数，需要多次询问",
                    "表达不清，使用模糊的描述",
                    "容易被复杂的技术术语吓到",
                    "需要手把手指导",
                    "经常问一些基础问题"
                ],
                communication_style="啰嗦而犹豫，经常使用'嗯...'，'这个...'等语气词",
                expertise_level="novice",
                patience_level="patient",
                complexity_threshold=1
            ),
            UserPersona(
                name="intermediate_user",
                description="普通用户，有一定技术基础但不专业",
                behavior_traits=[
                    "知道基本概念但经常记不清细节",
                    "会尝试自己解决问题但有时需要提示",
                    "关心效率，希望尽快完成任务",
                    "偶尔会提供不完整的参数"
                ],
                communication_style="直接而务实，使用日常口语",
                expertise_level="intermediate",
                patience_level="normal",
                complexity_threshold=2
            ),
            UserPersona(
                name="expert_user",
                description="专业用户，对API非常熟悉",
                behavior_traits=[
                    "提供准确的参数和要求",
                    "使用专业术语",
                    "关注细节和最佳实践",
                    "可能提供额外的优化建议"
                ],
                communication_style="简洁而精确，使用专业术语",
                expertise_level="expert",
                patience_level="impatient",
                complexity_threshold=5
            ),
            UserPersona(
                name="impatient_user",
                description="急躁的用户，希望快速解决问题",
                behavior_traits=[
                    "经常催促和表达不满",
                    "不愿意等待复杂的解释",
                    "优先选择最简单的解决方案",
                    "如果遇到问题可能会放弃"
                ],
                communication_style="简短而直接，带有命令语气",
                expertise_level="intermediate",
                patience_level="impatient",
                complexity_threshold=3
            ),
            UserPersona(
                name="confused_user",
                description="容易困惑的用户，面对复杂任务时会显得迷茫",
                behavior_traits=[
                    "经常需要重复确认",
                    "忘记之前说过的话",
                    "容易被过多信息 overwhelm",
                    "需要简单明了的指导"
                ],
                communication_style="犹豫不决，经常重复自己的问题",
                expertise_level="novice",
                patience_level="patient",
                complexity_threshold=1
            ),
            UserPersona(
                name="curious_user",
                description="好奇心强的用户，喜欢探索和提问",
                behavior_traits=[
                    "经常问为什么和怎么做",
                    "对技术细节感兴趣",
                    "可能会偏离主任务去探索相关功能",
                    "学习能力强但容易分心"
                ],
                communication_style="充满疑问，喜欢追根究底",
                expertise_level="intermediate",
                patience_level="patient",
                complexity_threshold=3
            )
        ]

    def select_persona_by_complexity(self, chain_length: int, tool_complexity: str = "medium") -> UserPersona:
        """
        根据工具链复杂度选择合适的用户画像

        参数:
            chain_length: 工具链长度
            tool_complexity: 工具复杂度 ("low", "medium", "high")

        返回:
            选择的UserPersona对象
        """
        # 计算综合复杂度分数
        complexity_score = chain_length

        # 根据工具复杂度调整分数
        if tool_complexity == "high":
            complexity_score += 2
        elif tool_complexity == "low":
            complexity_score -= 1

        # 确保分数在合理范围内
        complexity_score = max(1, min(5, complexity_score))

        # 选择合适的画像
        suitable_personas = []

        for persona in self.personas:
            if complexity_score <= persona.complexity_threshold:
                suitable_personas.append(persona)

        if suitable_personas:
            # 如果有多个合适的选择，随机选择一个
            import random
            return random.choice(suitable_personas)
        else:
            # 如果没有合适的，默认选择intermediate用户
            return self.get_persona_by_name("intermediate_user")

    def get_persona_by_name(self, name: str) -> UserPersona:
        """
        根据名称获取用户画像

        参数:
            name: 画像名称

        返回:
            UserPersona对象

        抛出:
            ValueError: 如果找不到指定名称的画像
        """
        for persona in self.personas:
            if persona.name == name:
                return persona

        raise ValueError(f"Persona '{name}' not found")

    def get_all_personas(self) -> List[UserPersona]:
        """
        获取所有可用的用户画像

        返回:
            UserPersona对象列表
        """
        return self.personas.copy()

    def add_persona(self, persona: UserPersona):
        """
        添加新的用户画像

        参数:
            persona: 要添加的UserPersona对象
        """
        self.personas.append(persona)

    def estimate_tool_complexity(self, tool_chain: List[str], tools: List) -> str:
        """
        估算工具链的复杂度

        参数:
            tool_chain: 工具名称列表
            tools: 工具定义列表

        返回:
            复杂度等级: "low", "medium", "high"
        """
        if not tool_chain:
            return "low"

        # 简单的复杂度估算逻辑
        chain_length = len(tool_chain)

        # 检查是否有复杂的工具类型
        complex_tools = ["query", "search", "analyze", "process"]
        has_complex_tools = any(
            any(complex_term in tool_name.lower() for complex_term in complex_tools)
            for tool_name in tool_chain
        )

        if chain_length >= 4 or has_complex_tools:
            return "high"
        elif chain_length >= 2:
            return "medium"
        else:
            return "low"


# 全局画像管理器实例
persona_manager = PersonaManager()


def get_persona_manager() -> PersonaManager:
    """获取全局画像管理器实例"""
    return persona_manager
