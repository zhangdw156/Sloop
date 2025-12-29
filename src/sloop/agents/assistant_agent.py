"""
Assistant Agent 实现
代表被测的AI系统，实现Chain of Thought逻辑
"""

import json
from typing import List, Dict, Any, Optional


class AssistantAgent:
    """
    Assistant Agent 类
    实现CoT逻辑：思考 -> 决策 -> 输出
    """

    def __init__(self, profile: Optional[Dict[str, Any]] = None, model_client=None):
        """
        初始化Assistant Agent

        Args:
            profile: Assistant的人设配置（可选）
            model_client: 模型客户端（预留，暂时未使用）
        """
        self.profile = profile or {
            "name": "Assistant",
            "personality": "helpful and intelligent",
            "capabilities": ["tool_calling", "conversation"]
        }
        self.model_client = model_client  # 预留给未来真实LLM调用

    def step(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        执行一步推理：思考 -> 决策 -> 输出

        Args:
            history: 对话历史列表

        Returns:
            包含thought、response_type和content的字典
        """
        # Thinking阶段：生成内心独白
        thought = self._generate_thought(history)

        # Decision阶段：判断是否需要工具调用
        needs_tool = self._decide_if_needs_tool(history)

        if needs_tool:
            # 生成工具调用
            tool_call = self._generate_tool_call(history)
            return {
                "thought": thought,
                "response_type": "tool_call",
                "content": tool_call
            }
        else:
            # 生成文本回复
            text_response = self._generate_text_response(history)
            return {
                "thought": thought,
                "response_type": "text",
                "content": text_response
            }

    def _generate_thought(self, history: List[Dict[str, Any]]) -> str:
        """
        生成思考内容（Mock实现）

        Args:
            history: 对话历史

        Returns:
            思考文本
        """
        if not history:
            return "这是对话的开始，我需要理解用户的需求。"

        latest_message = history[-1]
        user_content = latest_message.get("content", "").lower()

        # 根据内容生成不同的思考
        if "weather" in user_content:
            return "用户询问天气相关信息，我应该调用天气查询工具来获取准确数据。"
        elif "api" in user_content:
            return "用户提到了API，我需要使用相关的工具来处理这个请求。"
        elif any(word in user_content for word in ["hello", "hi", "你好"]):
            return "用户在打招呼，我应该友好地回应并询问如何帮助。"
        else:
            return "用户提出了一个一般性问题，我会直接给出有帮助的回答。"

    def _decide_if_needs_tool(self, history: List[Dict[str, Any]]) -> bool:
        """
        判断是否需要调用工具（Mock逻辑）

        Args:
            history: 对话历史

        Returns:
            是否需要工具调用
        """
        if not history:
            return False

        latest_message = history[-1]

        # 只有用户消息才可能触发工具调用
        if latest_message.get("role") != "user":
            return False

        user_content = latest_message.get("content", "").lower()

        # 简单的关键词触发逻辑
        tool_keywords = ["weather", "api", "查询", "获取", "查找"]
        return any(keyword in user_content for keyword in tool_keywords)

    def _generate_tool_call(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        生成工具调用（Mock实现）

        Args:
            history: 对话历史

        Returns:
            工具调用字典
        """
        latest_message = history[-1]
        user_content = latest_message.get("content", "").lower()

        if "weather" in user_content:
            return {
                "name": "get_weather",
                "arguments": {
                    "city": "Beijing",  # Mock默认城市
                    "unit": "celsius"
                }
            }
        elif "api" in user_content:
            return {
                "name": "search_api",
                "arguments": {
                    "query": user_content,
                    "limit": 5
                }
            }
        else:
            # 默认工具调用
            return {
                "name": "general_tool",
                "arguments": {
                    "query": user_content
                }
            }

    def _generate_text_response(self, history: List[Dict[str, Any]]) -> str:
        """
        生成文本回复（Mock实现）

        Args:
            history: 对话历史

        Returns:
            回复文本
        """
        latest_message = history[-1]
        user_content = latest_message.get("content", "").lower()

        if any(word in user_content for word in ["hello", "hi", "你好"]):
            return "你好！我是AI助手，很高兴为你服务。请告诉我你需要什么帮助？"
        elif "谢谢" in user_content or "thank" in user_content:
            return "不客气！如果还有其他问题，随时问我。"
        else:
            return f"我理解你的问题关于'{user_content}'。这是一个很好的问题，我会尽力帮助你。"


# Self-Check 测试代码
if __name__ == "__main__":
    # 初始化Assistant Agent
    agent = AssistantAgent()

    # 场景A: 工具调用场景
    print("=== 场景A: 工具调用场景 ===")
    history_a = [
        {"role": "user", "content": "help me check weather"}
    ]
    result_a = agent.step(history_a)
    print(f"Thought: {result_a['thought']}")
    print(f"Response Type: {result_a['response_type']}")
    print(f"Content: {json.dumps(result_a['content'], ensure_ascii=False, indent=2)}")

    print("\n" + "="*40)

    # 场景B: 普通对话场景
    print("=== 场景B: 普通对话场景 ===")
    history_b = [
        {"role": "user", "content": "hello"}
    ]
    result_b = agent.step(history_b)
    print(f"Thought: {result_b['thought']}")
    print(f"Response Type: {result_b['response_type']}")
    print(f"Content: {result_b['content']}")

    print("\n✅ Assistant Agent 测试完成")
