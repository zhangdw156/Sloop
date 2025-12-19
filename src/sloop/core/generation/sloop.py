"""
Sloop 主类，组织并协调所有 Agent，实现多智能体对话数据生成系统。
"""

from typing import Dict, Any, List
from sloop.core.config import SloopConfig
from sloop.core.generation.api_structure import APIStructure
from sloop.core.generation.sampler import APISampler
from sloop.core.agents.user_agent import UserAgent
from sloop.core.agents.assistant_agent import AssistantAgent
from sloop.core.agents.service_agent import ServiceAgent
from sloop.core.agents.planner import Planner
from sloop.core.agents.user_profile_agent import UserProfileAgent


class Sloop:
    """
    Sloop 类，作为多智能体系统的协调中心。
    它负责初始化所有 Agent，并驱动整个对话生成流程。
    """
    def __init__(
        self,
        config: SloopConfig,
        user_agent: UserAgent,
        assistant_agent: AssistantAgent,
        service_agent: ServiceAgent,
        planner: Planner,
        user_profile_agent: UserProfileAgent,
        api_sampler: APISampler,
    ):
        """
        初始化 Sloop 系统。
        
        Args:
            config (SloopConfig): Sloop 配置对象。
            user_agent (UserAgent): 用户代理实例。
            assistant_agent (AssistantAgent): 助手代理实例。
            service_agent (ServiceAgent): 服务代理实例。
            planner (Planner): 规划器实例。
            user_profile_agent (UserProfileAgent): 用户画像生成器实例。
            api_sampler (APISampler): API 采样器实例。
        """
        self.config = config
        self.user_agent = user_agent
        self.assistant_agent = assistant_agent
        self.service_agent = service_agent
        self.planner = planner
        self.user_profile_agent = user_profile_agent
        self.api_sampler = api_sampler

    def generate_conversation(self, problem: str, apis: List[APIStructure]) -> Dict[str, Any]:
        """
        生成一个完整的多轮对话。
        
        Args:
            problem (str): 用户需要解决的问题。
            apis (List[APIStructure]): 可用的 API 列表。
            
        Returns:
            Dict[str, Any]: 包含对话历史和最终标签的字典。
        """
        # 1. 生成用户画像
        user_profile = self.user_profile_agent.generate_profile(problem)
        
        # 2. 生成初始用户请求
        user_request = self.user_agent.generate_request(problem, user_profile)
        
        # 3. 规划对话流程
        plan = self.planner.plan_dialogue(problem, apis)
        
        # 4. 执行多轮对话
        conversation_history = [
            {"role": "user", "content": user_request}
        ]
        current_problem = problem
        
        for step in plan.get("steps", []):
            # 根据规划，调用相应的代理
            if step["type"] == "assistant":
                # 助手回复
                # 构建对话历史字符串
                history_str = "\n".join([
                    f"{msg['role']}: {msg['content']}"
                    for msg in conversation_history
                ])
                assistant_response = self.assistant_agent.respond(user_request, history_str)
                conversation_history.append({
                    "role": "assistant",
                    "content": assistant_response
                })
                # 更新用户请求为上一轮的回复
                user_request = assistant_response
            elif step["type"] == "service":  # type: ignore
                # 服务调用
                # 这里简化处理，直接使用规划中的工具调用
                tool_call = step["tool_call"]
                service_result = self.service_agent.execute_call(tool_call)
                conversation_history.append({
                    "role": "system",
                    "content": f"服务调用结果: {service_result}"
                })
                # 更新问题，可能需要根据服务结果调整
                current_problem = f"{current_problem} (服务调用后更新)"
        
        # 5. 构造最终的标签
        final_label = {
            "tool_call": plan.get("final_tool_call", {}),
            "thought_process": plan.get("thought_process", "")
        }
        
        return {
            "problem": problem,
            "conversation": conversation_history,
            "label": final_label
        }
