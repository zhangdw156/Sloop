"""
用户模拟器 (User Agent)

模拟用户行为，根据blueprint的意图和对话历史生成下一条用户消息。
"""

import logging
from typing import List, Optional
from ..models import Blueprint, ChatMessage
from ..utils.llm import chat_completion
from ..utils.template import render_user_prompt

logger = logging.getLogger(__name__)


class UserAgent:
    """
    用户智能体

    负责模拟用户行为，根据给定的意图和对话历史生成合理的用户消息。
    """

    def __init__(self):
        """初始化用户智能体"""
        logger.info("UserAgent initialized")

    def generate_message(
        self,
        blueprint: Blueprint,
        conversation_history: List[ChatMessage]
    ) -> str:
        """
        生成用户消息

        参数:
            blueprint: 对话蓝图，包含用户意图
            conversation_history: 对话历史消息列表

        返回:
            生成的用户消息字符串，如果任务完成则返回"###STOP###"
        """
        logger.info(f"Generating user message for intent: {blueprint.intent}")

        # 构造提示
        prompt = render_user_prompt(blueprint.intent, conversation_history)

        # 调用LLM生成消息
        response = chat_completion(
            prompt=prompt,
            system_message="You are a user in a conversation. Respond naturally and decide when the task is complete.",
            json_mode=False
        )

        if not response or response.startswith("调用错误"):
            logger.error(f"Failed to generate user message: {response}")
            return "I need help with something."  # 默认消息

        # 检查是否包含停止标记
        response = response.strip()
        if "###STOP###" in response:
            logger.info("User indicated task completion")
            return "###STOP###"

        logger.info(f"Generated user message: {response[:100]}...")
        return response

    def is_task_complete(self, message: str) -> bool:
        """
        检查任务是否完成

        参数:
            message: 用户消息

        返回:
            是否完成任务
        """
        return "###STOP###" in message


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("🤖 User Agent 测试")
    print("=" * 50)

    from ..models import Blueprint

    # 创建模拟blueprint
    mock_blueprint = Blueprint(
        intent="查找餐厅并点餐",
        required_tools=["find_restaurants", "get_menu"],
        ground_truth=["find_restaurants", "get_menu"],
        initial_state={"restaurant_found": False},
        expected_state={"restaurant_found": True, "menu_loaded": True}
    )

    # 创建模拟对话历史
    mock_history = [
        ChatMessage(role="assistant", content="你好！有什么可以帮助你的吗？"),
        ChatMessage(role="user", content="我想找一家餐厅吃饭"),
    ]

    print("📋 测试数据:")
    print(f"  意图: {mock_blueprint.intent}")
    print(f"  历史消息数: {len(mock_history)}")
    print()

    # 初始化用户智能体
    print("🔧 初始化UserAgent...")
    user_agent = UserAgent()

    print("💬 生成用户消息...")
    try:
        message = user_agent.generate_message(mock_blueprint, mock_history)

        print("✅ 生成成功！")
        print(f"📝 消息内容: {message}")

        if user_agent.is_task_complete(message):
            print("🎯 任务已完成")
        else:
            print("🔄 任务继续")

    except Exception as e:
        print(f"❌ 生成失败: {e}")

        # 如果LLM调用失败，提供模拟结果
        print("\n🔧 提供模拟用户消息:")
        print("我想在市中心找一家中餐厅。")

    print("\n✅ User Agent 测试完成！")
