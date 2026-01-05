"""
用户模拟器 (User Agent)

模拟用户行为，根据blueprint的意图和对话历史生成下一条用户消息。
"""

from typing import List

from sloop.models import Blueprint, ChatMessage
from sloop.utils.llm import chat_completion
from sloop.utils.logger import logger
from sloop.utils.template import render_user_prompt


class UserAgent:
    """
    用户智能体

    负责模拟用户行为，根据给定的意图和对话历史生成合理的用户消息。
    """

    def __init__(self):
        """初始化用户智能体"""
        logger.info("UserAgent initialized")

    def generate_message(
        self, blueprint: Blueprint, conversation_history: List[ChatMessage]
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

        # 记录persona信息
        if blueprint.persona:
            logger.info(
                f"Using persona: {blueprint.persona.name} ({blueprint.persona.description})"
            )

        # 构造提示
        prompt = render_user_prompt(
            blueprint.intent, conversation_history, blueprint.persona
        )

        # 调用LLM生成消息
        response = chat_completion(
            prompt=prompt,
            system_message="",
            json_mode=False,
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
    logger.info("🤖 User Agent 测试")
    logger.info("=" * 50)

    from sloop.models import Blueprint

    # 创建模拟blueprint
    mock_blueprint = Blueprint(
        intent="查找餐厅并点餐",
        required_tools=["find_restaurants", "get_menu"],
        ground_truth=["find_restaurants", "get_menu"],
        initial_state={"restaurant_found": False},
        expected_state={"restaurant_found": True, "menu_loaded": True},
    )

    # 创建模拟对话历史
    mock_history = [
        ChatMessage(role="assistant", content="你好！有什么可以帮助你的吗？"),
        ChatMessage(role="user", content="我想找一家餐厅吃饭"),
    ]

    logger.info("📋 测试数据:")
    logger.info(f"  意图: {mock_blueprint.intent}")
    logger.info(f"  历史消息数: {len(mock_history)}")
    logger.info("")

    # 初始化用户智能体
    logger.info("🔧 初始化UserAgent...")
    user_agent = UserAgent()

    logger.info("💬 生成用户消息...")
    try:
        message = user_agent.generate_message(mock_blueprint, mock_history)

        logger.info("✅ 生成成功！")
        logger.info(f"📝 消息内容: {message}")

        if user_agent.is_task_complete(message):
            logger.info("🎯 任务已完成")
        else:
            logger.info("🔄 任务继续")

    except Exception as e:
        logger.error(f"❌ 生成失败: {e}")

        # 如果LLM调用失败，提供模拟结果
        logger.info("\n🔧 提供模拟用户消息:")
        logger.info("我想在市中心找一家中餐厅。")

    logger.info("\n✅ User Agent 测试完成！")
