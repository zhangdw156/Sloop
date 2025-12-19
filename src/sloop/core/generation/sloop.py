"""
Sloop 主类，组织并协调所有 Agent，实现多智能体对话数据生成系统。
"""

from typing import Any, Dict, List

from sloop.core.config import SloopConfig
from sloop.core.generation.api_structure import APIStructure


class Sloop:
    """
    Sloop 类，作为多智能体系统的协调中心。
    它负责初始化所有 Agent，并驱动整个对话生成流程。
    """

    def __init__(self, agents: Dict[str, Any], config: SloopConfig):
        """
        初始化 Sloop 系统。

        Args:
            agents (Dict[str, Any]): 包含所有必要代理的字典。
            config (SloopConfig): Sloop 配置对象。
        """
        self.config = config
        self.user_agent = agents["user_agent"]
        self.assistant_agent = agents["assistant_agent"]
        self.service_agent = agents["service_agent"]
        self.planner = agents["planner"]
        self.user_profile_agent = agents["user_profile_agent"]
        self.api_sampler = agents["api_sampler"]

    def generate_conversation(
        self, problem: str, apis: List[APIStructure]
    ) -> Dict[str, Any]:
        """
        生成一个完整的多轮对话。

        Args:
            problem (str): 用户需要解决的问题。
            apis (List[APIStructure]): 可用的 API 列表。

        Returns:
            Dict[str, Any]: 包含对话历史和最终标签的字典。
        """
        # 1. 生成用户画像
        context = {"problem": problem, "apis": list(apis)}
        user_profile = self.user_profile_agent.generate_profile(context)

        # 2. 生成初始用户请求
        user_request = self.user_agent.generate_request(problem, user_profile)

        # 3. 规划对话流程
        plan = self.planner.plan_dialogue(problem, apis)

        # 4. 执行多轮对话
        # 对话顺序: User -> Assistant (生成 Label) -> Tool (System/Observation) -> Assistant (总结)
        conversation_history = [{"role": "user", "content": user_request}]

        # 第一轮: 助手生成标签和思考过程
        # 构建对话历史字符串
        history_str = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in conversation_history
        ])
        first_assistant_response = self.assistant_agent.respond(
            user_request, history_str
        )
        conversation_history.append({
            "role": "assistant",
            "content": first_assistant_response,
        })

        # 从第一轮助手回复中提取标签和工具调用
        thought_process = ""
        tool_call = {}
        # 使用正则表达式提取 `<tool_call>` 标签内容作为思考过程
        import re
        thought_match = re.search(r'<tool_call>(.*?)<tool_call>', first_assistant_response)
        if thought_match:
            thought_process = thought_match.group(1)

        # 使用正则表达式提取 `<tool_call>` 标签内容并解析为 JSON 作为工具调用
        tool_match = re.search(r'<tool_call>(.*?)<tool_call>', first_assistant_response)
        if tool_match:
            import json
            try:
                tool_call = json.loads(tool_match.group(1))
            except json.JSONDecodeError:
                # 如果解析失败，保留原始字符串或设置默认值
                tool_call = {"name": "解析失败", "arguments": {}}

        # 第二轮: 服务调用
        # 使用提取到的 tool_call
        service_result = self.service_agent.execute_call(tool_call)
        conversation_history.append({
            "role": "tool",  # 将角色从 `system` 更改为 `tool`
            "content": f"服务调用结果: {service_result}",
        })

        # 第三轮: 助手生成最终总结
        # 更新对话历史字符串
        history_str = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in conversation_history
        ])
        final_assistant_response = self.assistant_agent.respond(
            user_request, history_str
        )
        conversation_history.append({
            "role": "assistant",
            "content": final_assistant_response,
        })

        # 5. 构造最终的标签
        final_label = {
            "tool_call": tool_call,
            "thought_process": thought_process,
        }

        return {
            "problem": problem,
            "conversation": conversation_history,
            "label": final_label,
        }
