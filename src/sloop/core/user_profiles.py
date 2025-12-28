"""
用户画像生成模块
创建多样化的用户类型，用于生成更真实的对话场景
"""

from typing import Dict, Any, List
from enum import Enum
import random


class UserType(Enum):
    """用户类型枚举"""
    CAREFUL = "careful"          # 细心的用户 - 准确表达需求
    CARELESS = "careless"        # 粗心的用户 - 经常给错参数
    UNCLEAR = "unclear"          # 表达不清的用户 - 需求表达不准确
    CURIOUS = "curious"          # 好奇的用户 - 喜欢探索和提问
    TECHNICAL = "technical"      # 技术型用户 - 关注技术细节
    BUSINESS = "business"        # 商务用户 - 关注效率和结果
    NOVICE = "novice"           # 新手用户 - 不熟悉功能


class UserProfileAgent:
    """
    用户画像生成器
    创建不同类型的用户，用于生成多样化的对话场景
    """

    def __init__(self):
        self.user_types = {
            UserType.CAREFUL: self._create_careful_profile,
            UserType.CARELESS: self._create_careless_profile,
            UserType.UNCLEAR: self._create_unclear_profile,
            UserType.CURIOUS: self._create_curious_profile,
            UserType.TECHNICAL: self._create_technical_profile,
            UserType.BUSINESS: self._create_business_profile,
            UserType.NOVICE: self._create_novice_profile,
        }

    def generate_profile(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        生成用户画像

        Args:
            context: 生成上下文（可选）

        Returns:
            用户画像字典
        """
        # 根据上下文选择用户类型，或者随机选择
        if context and 'preferred_user_type' in context:
            user_type = UserType(context['preferred_user_type'])
        else:
            user_type = random.choice(list(UserType))

        # 生成基础画像
        profile = self.user_types[user_type]()

        # 添加上下文相关的特征
        if context:
            profile = self._adapt_to_context(profile, context)

        return profile

    def generate_profiles_batch(self, count: int, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        批量生成用户画像

        Args:
            count: 生成数量
            context: 生成上下文

        Returns:
            用户画像列表
        """
        profiles = []
        for _ in range(count):
            profile = self.generate_profile(context)
            profiles.append(profile)
        return profiles

    def _create_careful_profile(self) -> Dict[str, Any]:
        """创建细心用户的画像"""
        return {
            "type": "careful",
            "name": random.choice(["小明", "小红", "小刚", "小丽", "小王"]),
            "age": random.randint(25, 45),
            "occupation": random.choice(["工程师", "设计师", "教师", "医生", "研究员"]),
            "personality": "细心、耐心、善于规划",
            "communication_style": "清晰、准确、有条理",
            "error_handling": "很少出错，但会仔细检查结果",
            "interaction_pattern": "一次性提供完整准确的信息",
            "typical_behavior": "先思考后行动，喜欢确认细节"
        }

    def _create_careless_profile(self) -> Dict[str, Any]:
        """创建粗心用户的画像"""
        return {
            "type": "careless",
            "name": random.choice(["小李", "小张", "小赵", "小孙", "小周"]),
            "age": random.randint(18, 35),
            "occupation": random.choice(["学生", "自由职业", "销售", "客服"]),
            "personality": "马虎、急躁、容易分心",
            "communication_style": "快速、简略、经常遗漏细节",
            "error_handling": "经常给错参数，发现问题后才会修正",
            "interaction_pattern": "多轮对话，逐步修正错误",
            "typical_behavior": "着急完成任务，参数经常输错",
            "common_mistakes": ["拼写错误", "参数顺序错", "单位错", "格式错"]
        }

    def _create_unclear_profile(self) -> Dict[str, Any]:
        """创建表达不清用户的画像"""
        return {
            "type": "unclear",
            "name": random.choice(["小陈", "小吴", "小郑", "小冯", "小蒋"]),
            "age": random.randint(35, 60),
            "occupation": random.choice(["经理", "主管", "家长", "退休人员"]),
            "personality": "犹豫、保守、不善于表达",
            "communication_style": "模糊、委婉、需要引导",
            "error_handling": "表达不清，需要助手多次确认",
            "interaction_pattern": "多次澄清需求，多轮对话",
            "typical_behavior": "用模糊语言表达，喜欢说'大概'、'差不多'",
            "communication_habits": ["用代词", "省略主语", "模糊描述", "需要确认"]
        }

    def _create_curious_profile(self) -> Dict[str, Any]:
        """创建好奇用户的画像"""
        return {
            "type": "curious",
            "name": random.choice(["小林", "小杨", "小刘", "小黄", "小徐"]),
            "age": random.randint(20, 40),
            "occupation": random.choice(["记者", "研究员", "教师", "学生"]),
            "personality": "好奇、探索、求知欲强",
            "communication_style": "提问多、探索性、喜欢了解细节",
            "error_handling": "通过提问发现和解决问题",
            "interaction_pattern": "主动提问，深入探索功能",
            "typical_behavior": "问很多'为什么'和'怎么做'，喜欢尝试新功能",
            "exploration_style": ["问功能细节", "尝试边界情况", "比较不同选项"]
        }

    def _create_technical_profile(self) -> Dict[str, Any]:
        """创建技术型用户的画像"""
        return {
            "type": "technical",
            "name": random.choice(["小郭", "小何", "小宋", "小唐", "小韩"]),
            "age": random.randint(25, 50),
            "occupation": random.choice(["程序员", "架构师", "数据科学家", "系统管理员"]),
            "personality": "严谨、技术导向、追求完美",
            "communication_style": "专业、精确、使用技术术语",
            "error_handling": "关注错误码和异常情况",
            "interaction_pattern": "关心技术实现细节和边界情况",
            "typical_behavior": "问API参数格式，关心错误处理，测试边界情况",
            "technical_focus": ["API规范", "错误处理", "性能", "安全性"]
        }

    def _create_business_profile(self) -> Dict[str, Any]:
        """创建商务用户的画像"""
        return {
            "type": "business",
            "name": random.choice(["李总", "王经理", "张主管", "刘总监", "陈经理"]),
            "age": random.randint(35, 55),
            "occupation": random.choice(["总经理", "部门经理", "项目经理", "企业主"]),
            "personality": "务实、效率导向、结果驱动",
            "communication_style": "简洁、直接、关注ROI",
            "error_handling": "关心影响效率的问题",
            "interaction_pattern": "快速完成任务，关注业务价值",
            "typical_behavior": "问'能不能'、'要多久'、'有什么用'",
            "business_focus": ["效率", "成本", "效果", "可行性"]
        }

    def _create_novice_profile(self) -> Dict[str, Any]:
        """创建新手用户的画像"""
        return {
            "type": "novice",
            "name": random.choice(["小白", "新用户", "小菜", "萌新"]),
            "age": random.randint(18, 30),
            "occupation": random.choice(["学生", "应届毕业生", "新员工"]),
            "personality": "迷茫、需要指导、学习态度好",
            "communication_style": "简单、基础、经常问'怎么做'",
            "error_handling": "不知道怎么处理错误，需要手把手指导",
            "interaction_pattern": "需要详细解释，多轮确认",
            "typical_behavior": "问基本概念，担心出错，寻求确认",
            "learning_style": ["需要示例", "喜欢步骤指导", "担心犯错"]
        }

    def _adapt_to_context(self, profile: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据上下文调整用户画像

        Args:
            profile: 原始画像
            context: 上下文信息

        Returns:
            调整后的画像
        """
        # 根据API类型调整画像
        if 'apis' in context:
            apis = context['apis']
            api_categories = set()
            for api in apis:
                category = api.get('category', 'general')
                api_categories.add(category)

            # 如果是技术相关的API，更倾向于技术型用户
            if 'search' in api_categories or 'data' in api_categories:
                if profile['type'] != 'technical':
                    profile['technical_interest'] = "对数据和搜索感兴趣"

            # 如果是商务相关的API，更倾向于商务用户
            if 'finance' in api_categories:
                if profile['type'] != 'business':
                    profile['business_focus'] = "关注财务和商务数据"

        return profile


class UserBehaviorSimulator:
    """
    用户行为模拟器
    根据用户画像生成具体的对话行为
    """

    def __init__(self, profile: Dict[str, Any]):
        self.profile = profile
        self.user_type = UserType(profile['type'])

    def generate_initial_request(self, problem: str, apis: List[Dict[str, Any]]) -> str:
        """
        根据用户画像生成初始请求

        Args:
            problem: 基础问题
            apis: 可用API列表

        Returns:
            生成的用户请求
        """
        if self.user_type == UserType.CAREFUL:
            return self._careful_request(problem, apis)
        elif self.user_type == UserType.CARELESS:
            return self._careless_request(problem, apis)
        elif self.user_type == UserType.UNCLEAR:
            return self._unclear_request(problem, apis)
        elif self.user_type == UserType.CURIOUS:
            return self._curious_request(problem, apis)
        elif self.user_type == UserType.TECHNICAL:
            return self._technical_request(problem, apis)
        elif self.user_type == UserType.BUSINESS:
            return self._business_request(problem, apis)
        elif self.user_type == UserType.NOVICE:
            return self._novice_request(problem, apis)
        else:
            return problem

    def _careful_request(self, problem: str, apis: List[Dict[str, Any]]) -> str:
        """细心用户的请求 - 准确、详细"""
        api_names = [api['name'] for api in apis[:2]]  # 只提最相关的
        return f"我想{problem}，我需要使用{api_names[0]}功能，请帮我操作。"

    def _careless_request(self, problem: str, apis: List[Dict[str, Any]]) -> str:
        """粗心用户的请求 - 参数经常错"""
        # 故意在参数上出错
        if "北京" in problem:
            return f"我想{problem.replace('北京', '上海')}"  # 城市搞错
        return f"我想{problem}，快点帮我搞定！"

    def _unclear_request(self, problem: str, apis: List[Dict[str, Any]]) -> str:
        """表达不清的请求 - 模糊、需要澄清"""
        return f"我想{problem}，大概就是那个东西，你知道吧？"

    def _curious_request(self, problem: str, apis: List[Dict[str, Any]]) -> str:
        """好奇用户的请求 - 喜欢探索"""
        return f"我想{problem}，这个功能有什么特别的地方吗？可以怎么用？"

    def _technical_request(self, problem: str, apis: List[Dict[str, Any]]) -> str:
        """技术用户的请求 - 关注技术细节"""
        api_names = [api['name'] for api in apis[:3]]
        return f"我想{problem}，请使用{', '.join(api_names)}API，注意参数格式和错误处理。"

    def _business_request(self, problem: str, apis: List[Dict[str, Any]]) -> str:
        """商务用户的请求 - 关注效率"""
        return f"我想{problem}，请高效完成，最好快一点。"

    def _novice_request(self, problem: str, apis: List[Dict[str, Any]]) -> str:
        """新手用户的请求 - 基础、需要指导"""
        return f"我想{problem}，但我不太会用，这个怎么操作啊？"


# 全局用户画像生成器实例
user_profile_agent = UserProfileAgent()
