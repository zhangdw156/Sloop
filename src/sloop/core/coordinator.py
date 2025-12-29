"""
Coordinator 实现
多智能体对话仿真协调器
"""

import json
from typing import List, Dict, Any
from sloop.agents.user_agent import UserAgent
from sloop.agents.assistant_agent import AssistantAgent
from sloop.agents.service_agent import ServiceAgent


class Coordinator:
    """
    Coordinator 类
    负责协调多智能体间的对话仿真
    """

    def __init__(self, user_profile: Dict[str, Any], assistant_profile: Dict[str, Any], tools_schema: List[Dict[str, Any]]):
        """
        初始化Coordinator

        Args:
            user_profile: 用户画像
            assistant_profile: Assistant人设
            tools_schema: 工具定义列表
        """
        self.user_profile = user_profile
        self.assistant_profile = assistant_profile
        self.tools_schema = tools_schema

        # 初始化各个Agent
        self.user_agent = UserAgent(user_profile)
        self.assistant_agent = AssistantAgent(assistant_profile)
        self.service_agent = ServiceAgent(tools_schema)

        # 全局对话历史
        self.history = []

    def run(self, max_turns: int = 5):
        """
        运行对话仿真

        Args:
            max_turns: 最大对话轮数
        """
        print("🚀 开始多智能体对话仿真")
        print(f"📋 用户风格: {self.user_profile.get('style', 'general')}")
        print(f"🤖 Assistant 配置: {self.assistant_profile.get('name', 'Assistant')}")
        print(f"🔧 可用工具数量: {len(self.tools_schema)}")
        print("=" * 50)

        turn = 1
        while turn <= max_turns:
            print(f"\n🔄 第 {turn} 轮对话")
            print("-" * 30)

            # === User Turn ===
            print("👤 用户发言:")
            user_message = self.user_agent.speak(self.history)
            user_msg = {"role": "user", "content": user_message}
            self.history.append(user_msg)
            print(f"   {user_message}")

            # === Assistant Turn (可能多次迭代) ===
            assistant_responding = True
            while assistant_responding:
                print("\n🤖 Assistant 思考中...")

                # Assistant 执行一步推理
                result = self.assistant_agent.step(self.history)

                thought = result["thought"]
                response_type = result["response_type"]
                content = result["content"]

                if response_type == "text":
                    # 文本回复：格式化为最终回复
                    final_content = f"<think>\n{thought}\n</think>\n\n{content}"
                    assistant_msg = {"role": "assistant", "content": final_content}
                    self.history.append(assistant_msg)

                    print("🤖 Assistant 回复:")
                    print(f"   {final_content}")
                    assistant_responding = False  # 结束Assistant turn

                elif response_type == "tool_call":
                    # 工具调用：格式化并添加到历史
                    tool_str = json.dumps(content, ensure_ascii=False)
                    tool_content = f"<think>\n{thought}\n</think>\n\n<tool_call>{tool_str}</tool_call>"
                    tool_call_msg = {"role": "assistant", "content": tool_content}
                    self.history.append(tool_call_msg)

                    print("🤖 Assistant 发起工具调用:")
                    print(f"   工具: {content.get('name', 'unknown')}")
                    print(f"   参数: {json.dumps(content.get('arguments', {}), ensure_ascii=False, indent=2)}")

                    # === Service Execution ===
                    print("\n🔧 Service 执行工具...")
                    tool_name = content.get("name", "")
                    tool_args = content.get("arguments", {})

                    observation = self.service_agent.execute(tool_name, tool_args)
                    tool_result_msg = {"role": "tool", "content": observation}
                    self.history.append(tool_result_msg)

                    print("🔧 Service 返回结果:")
                    print(f"   {observation}")

                    # Assistant 继续思考（不结束turn）
                    print("🤖 Assistant 继续处理结果...")
                    assistant_responding = True

            # 轮次结束
            turn += 1
            print(f"\n✅ 第 {turn-1} 轮完成")

        print("\n🎉 对话仿真结束")
        print(f"📊 总消息数: {len(self.history)}")
        print("=" * 50)


# End-to-End 测试代码
if __name__ == "__main__":
    print("🧪 Coordinator End-to-End 测试")

    # 定义测试配置
    user_profile = {
        "name": "TestUser",
        "style": "curious"
    }

    assistant_profile = {
        "name": "TestAssistant",
        "personality": "helpful and intelligent",
        "capabilities": ["tool_calling", "conversation"]
    }

    tools_schema = [
        {
            "name": "get_weather",
            "description": "获取天气信息",
            "parameters": {"city": "string", "unit": "string"}
        },
        {
            "name": "search_info",
            "description": "搜索信息",
            "parameters": {"query": "string"}
        }
    ]

    # 初始化Coordinator
    coordinator = Coordinator(user_profile, assistant_profile, tools_schema)

    # 运行仿真
    coordinator.run(max_turns=3)

    print("\n✅ End-to-End 测试完成")
