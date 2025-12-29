"""
User Agent 实现
模拟用户行为，根据人设发起对话或回复Assistant
"""

from typing import List, Dict, Any
import random


class UserAgent:
    """
    User Agent 类
    负责模拟用户对话行为
    """

    def __init__(self, profile: Dict[str, Any]):
        """
        初始化User Agent

        Args:
            profile: 用户画像字典，包含name, style等信息
        """
        self.profile = profile
        self.name = profile.get("name", "User")
        self.style = profile.get("style", "general")

        # 根据style预定义不同类型的开场白和回复
        self.opening_lines = {
            "curious": [
                "Can you help me understand how APIs work?",
                "I'm curious about weather forecasting. How does it work?",
                "What can you tell me about machine learning?"
            ],
            "technical": [
                "I need to integrate with a weather API. Can you show me how?",
                "What's the best way to handle API rate limits?",
                "How do I parse JSON responses from REST APIs?"
            ],
            "casual": [
                "Hey, what's the weather like today?",
                "Can you check something for me?",
                "I have a question about APIs"
            ],
            "general": [
                "Hello! Can you help me with something?",
                "I need assistance with a task",
                "Can you tell me about weather information?"
            ]
        }

        self.followup_responses = {
            "curious": [
                "That's interesting! Can you tell me more?",
                "How does that work exactly?",
                "What are the limitations of this approach?"
            ],
            "technical": [
                "What are the technical specifications?",
                "How do I implement this in code?",
                "Are there any best practices I should follow?"
            ],
            "casual": [
                "Thanks! That's helpful.",
                "Cool, I appreciate it.",
                "Got it, thanks for the info."
            ],
            "general": [
                "Thank you for the information.",
                "I understand now.",
                "That makes sense."
            ]
        }

    def speak(self, history: List[Dict[str, Any]]) -> str:
        """
        根据对话历史生成用户发言

        Args:
            history: 当前对话历史

        Returns:
            用户的发言内容（字符串）
        """
        if not history:
            # 对话开始，生成开场白
            return self._generate_opening_line()
        else:
            # 对话进行中，生成后续回复
            return self._generate_followup_response(history)

    def _generate_opening_line(self) -> str:
        """
        生成开场白

        Returns:
            开场白字符串
        """
        style = self.style
        if style not in self.opening_lines:
            style = "general"

        lines = self.opening_lines[style]
        return random.choice(lines)

    def _generate_followup_response(self, history: List[Dict[str, Any]]) -> str:
        """
        生成后续回复（Mock实现）

        Args:
            history: 对话历史

        Returns:
            后续回复字符串
        """
        # 简单的Mock逻辑：随机选择回复，但可以根据Assistant的回复内容调整
        last_assistant_message = None
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                last_assistant_message = msg
                break

        style = self.style
        if style not in self.followup_responses:
            style = "general"

        responses = self.followup_responses[style]

        # 如果Assistant的消息包含某些关键词，可以调整回复
        if last_assistant_message:
            content = last_assistant_message.get("content", "").lower()
            if "weather" in content or "api" in content:
                # 如果Assistant提到了天气或API，显示更感兴趣
                if style == "curious":
                    return "That's fascinating! How does the weather API handle different locations?"
                elif style == "technical":
                    return "What are the API endpoints for weather data?"
                else:
                    return "Thanks for the weather info! That's really helpful."

        return random.choice(responses)


# Self-Check 测试代码
if __name__ == "__main__":
    # 创建不同风格的用户画像
    profiles = [
        {"name": "Alice", "style": "curious"},
        {"name": "Bob", "style": "technical"},
        {"name": "Charlie", "style": "casual"}
    ]

    for profile in profiles:
        print(f"\n=== 测试用户: {profile['name']} ({profile['style']} 风格) ===")
        agent = UserAgent(profile)

        # 场景1: 开场发言
        print("场景1 (开场):")
        opening = agent.speak([])
        print(f"  用户说: {opening}")

        # 场景2: 后续回复
        print("场景2 (后续回复):")
        # 模拟包含Assistant回复的历史
        history_with_assistant = [
            {"role": "user", "content": opening},
            {"role": "assistant", "content": "I'd be happy to help you with that! Let me check the weather API for you."}
        ]
        followup = agent.speak(history_with_assistant)
        print(f"  用户说: {followup}")

    print("\n✅ User Agent 测试完成")
