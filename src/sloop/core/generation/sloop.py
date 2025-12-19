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
        # 修正对话顺序: User -> Assistant (生成思考和工具调用) -> Tool (返回执行结果) -> Assistant (生成最终总结)
        conversation_history = [{"role": "user", "content": user_request}]

        # 第一轮: 助手根据用户请求生成包含思考和工具调用的响应
        first_assistant_response = self.assistant_agent.respond(
            user_request, conversation_history
        )
        conversation_history.append({
            "role": "assistant",
            "content": first_assistant_response,
        })

        # 从第一轮助手的回复中提取标签信息
        thought_process = ""
        tool_call = {}
        # 使用正则表达式提取 `<tool_call>` 标签内容作为思考过程
        import re
        # 修正：使用更精确的正则表达式，确保匹配第一个 `<tool_call>...<tool_call>` 块作为思考过程
        thought_pattern = r'<tool_call>(.*?)<tool_call>'
        thought_matches = re.findall(thought_pattern, first_assistant_response, re.DOTALL)
        if thought_matches:
            # 取第一个匹配项作为思考过程
            thought_process = thought_matches[0].strip()

        # 使用正则表达式提取 `<tool_call>` 标签内容并解析为 JSON 作为工具调用
        # 修正：使用更精确的正则表达式，确保匹配第二个 `<tool_call>...<tool_call>` 块作为工具调用
        tool_pattern = r'<tool_call>(.*?)</tool_call>'
        tool_matches = re.findall(tool_pattern, first_assistant_response, re.DOTALL)
        if len(tool_matches) > 1:
            # 取第二个匹配项作为工具调用
            tool_json_str = tool_matches[1].strip()
            try:
                tool_call = json.loads(tool_json_str)
            except json.JSONDecodeError as e:
                print(f"JSON 解析失败: {e}, 字符串: {tool_json_str}")
                tool_call = {"name": "解析失败", "arguments": {}}
        else:
            print(f"未在助手回复中找到足够的 `<tool_call>...</tool_call>` 模式。找到 {len(tool_matches)} 个。")
            tool_call = {"name": "解析失败", "arguments": {}}

        # 第二轮: 服务代理执行工具调用
        service_result = self.service_agent.execute_call(tool_call)
        conversation_history.append({
            "role": "tool", # 已正确使用 "tool" 角色
            "content": service_result, # 修正：直接使用 service_result，无需额外前缀
        })

        # 第三轮: 助手根据工具返回结果生成最终的自然语言总结
        final_assistant_response = self.assistant_agent.respond(
            user_request, conversation_history
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
