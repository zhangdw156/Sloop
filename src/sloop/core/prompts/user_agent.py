"""
用户代理（User Agent）提示词模板。
"""

# 用于生成初始用户请求的提示词
GENERATE_REQUEST_PROMPT = """
基于以下问题和用户画像，生成一个用户的初始请求:
问题: {problem}
用户画像: {user_profile}
""".strip()

# 用于根据问题和上下文生成用户消息的提示词
GENERATE_USER_MESSAGE_PROMPT = """
基于以下问题和上下文，生成一个用户的初始请求:
问题: {problem}
上下文: {context}
""".strip()
