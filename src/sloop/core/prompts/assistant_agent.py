"""
助手代理（Assistant Agent）提示词模板。
"""

# 用于生成助手回复的提示词
GENERATE_RESPONSE_PROMPT = """
你是一个助手，请根据以下对话历史和用户的新消息进行回复。
对话历史: {conversation_history}
用户: {user_message}
""".strip()
