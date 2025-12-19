"""
助手代理（Assistant Agent）提示词模板。
"""

# 用于生成助手回复的提示词
# 当对话历史中只有用户消息时，助手必须输出思考过程和工具调用
GENERATE_RESPONSE_PROMPT_FOR_TOOL_CALL = """
你是一个智能助手。当用户提出请求时，你需要：
1. 首先用 <tool_call>... found 标签包裹你的推理过程。
2. 然后用 <tool_call>...<tool_call> 标签包裹一个有效的 JSON，该 JSON 描述了你将调用的工具及其参数。
对话历史: {conversation_history}
用户: {user_message}
""".strip()

# 用于生成助手回复的提示词
# 当对话历史中包含 tool 角色的消息时，助手必须只输出最终的自然语言总结
GENERATE_RESPONSE_PROMPT_FOR_SUMMARY = """
你是一个智能助手。根据工具返回的执行结果，请用自然语言向用户总结最终结果。
请不要包含任何 XML 标签或 JSON。
对话历史: {conversation_history}
""".strip()
