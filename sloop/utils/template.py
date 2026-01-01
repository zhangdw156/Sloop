"""
统一模板管理模块

负责所有Jinja2模板的加载和渲染。
"""

import os

from jinja2 import Template


def _get_template_path(template_name: str) -> str:
    """
    获取模板文件路径

    参数:
        template_name: 模板文件名（不含扩展名）

    返回:
        模板文件的完整路径
    """
    # 从utils目录向上查找templates目录
    utils_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(utils_dir)  # src/sloop
    template_path = os.path.join(project_root, "templates", f"{template_name}.j2")
    return template_path


def _load_template(template_name: str) -> Template:
    """
    加载模板文件

    参数:
        template_name: 模板文件名（不含扩展名）

    返回:
        编译后的Jinja2模板对象
    """
    template_path = _get_template_path(template_name)

    with open(template_path, "r", encoding="utf-8") as f:
        template_content = f.read()

    return Template(template_content)


def get_planner_template() -> Template:
    """
    获取蓝图规划器的Jinja2模板

    返回:
        编译后的Jinja2模板对象
    """
    return _load_template("planner")


def render_planner_prompt(tool_chain: list, tool_definitions: list) -> str:
    """
    渲染蓝图规划器提示

    参数:
        tool_chain: 工具调用链列表
        tool_definitions: 工具定义列表

    返回:
        渲染后的提示字符串
    """
    template = get_planner_template()

    # 将ToolDefinition对象转换为字典格式，以便JSON序列化
    tool_definitions_dict = []
    for tool in tool_definitions:
        tool_dict = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters.model_dump()
            if hasattr(tool.parameters, "model_dump")
            else tool.parameters,
        }
        tool_definitions_dict.append(tool_dict)

    return template.render(
        tool_chain=tool_chain, tool_definitions=tool_definitions_dict
    )


def get_user_template():
    """
    获取用户智能体模板

    返回:
        编译后的Jinja2模板对象
    """
    return _load_template("user")


def render_user_prompt(intent: str, conversation_history: list) -> str:
    """
    渲染用户智能体提示

    参数:
        intent: 用户意图
        conversation_history: 对话历史消息列表

    返回:
        渲染后的提示字符串
    """
    template = get_user_template()

    # 转换消息对象为字典格式（兼容对象和字典）
    history_dict = []
    for message in conversation_history:
        if hasattr(message, "role") and hasattr(message, "content"):
            # 这是ChatMessage对象
            msg_dict = {"role": message.role, "content": message.content}
        else:
            # 这已经是字典了
            msg_dict = message
        history_dict.append(msg_dict)

    return template.render(intent=intent, conversation_history=history_dict)


def get_service_template():
    """
    获取服务智能体模板

    返回:
        编译后的Jinja2模板对象
    """
    return _load_template("service")


def render_service_prompt(
    tool_call, current_state, blueprint, conversation_history=None
) -> str:
    """
    渲染服务智能体提示

    参数:
        tool_call: 工具调用对象
        current_state: 当前状态对象
        blueprint: 蓝图对象
        conversation_history: 对话历史消息列表（可选）

    返回:
        渲染后的提示字符串
    """
    template = get_service_template()

    # 转换对象为字典格式（兼容对象和字典）
    if hasattr(tool_call, "name") and hasattr(tool_call, "arguments"):
        # 这是ToolCall对象
        tool_call_dict = {"tool_name": tool_call.name, "arguments": tool_call.arguments}
    else:
        # 这已经是字典了
        tool_call_dict = tool_call

    state_dict = (
        current_state.model_dump()
        if hasattr(current_state, "model_dump")
        else current_state.__dict__
    )

    blueprint_dict = {
        "intent": blueprint.intent,
        "expected_state": blueprint.expected_state,
    }

    # 转换对话历史为字典格式
    history_dict = []
    if conversation_history:
        for message in conversation_history:
            if hasattr(message, "role") and hasattr(message, "content"):
                # 这是ChatMessage对象
                msg_dict = {"role": message.role, "content": message.content}
            else:
                # 这已经是字典了
                msg_dict = message
            history_dict.append(msg_dict)

    return template.render(
        tool_call=tool_call_dict,
        current_state=state_dict,
        blueprint=blueprint_dict,
        conversation_history=history_dict,
    )


def get_assistant_think_template():
    """
    获取助手思考模板

    返回:
        编译后的Jinja2模板对象
    """
    return _load_template("assistant_think")


def render_assistant_think_prompt(
    conversation_history: list, context_hint: str = ""
) -> str:
    """
    渲染助手思考提示

    参数:
        conversation_history: 对话历史消息列表
        context_hint: 栈上下文提示信息（可选）

    返回:
        渲染后的提示字符串
    """
    template = get_assistant_think_template()

    # 转换消息对象为字典格式
    history_dict = []
    for message in conversation_history:
        msg_dict = {"role": message.role, "content": message.content}
        history_dict.append(msg_dict)

    return template.render(conversation_history=history_dict, context_hint=context_hint)


def get_assistant_decide_template():
    """
    获取助手决策模板

    返回:
        编译后的Jinja2模板对象
    """
    return _load_template("assistant_decide")


def render_assistant_decide_prompt(thought: str, tools: list) -> str:
    """
    渲染助手决策提示

    参数:
        thought: 思考过程字符串
        tools: 工具定义列表

    返回:
        渲染后的提示字符串
    """
    template = get_assistant_decide_template()

    # 转换工具对象为字典格式
    tools_dict = []
    for tool in tools:
        tool_dict = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters.model_dump()
            if hasattr(tool.parameters, "model_dump")
            else tool.parameters,
        }
        tools_dict.append(tool_dict)

    return template.render(thought=thought, tools=tools_dict)


def get_tool_call_gen_template():
    """
    获取工具调用生成模板

    返回:
        编译后的Jinja2模板对象
    """
    return _load_template("tool_call_gen")


def render_tool_call_gen_prompt(thought: str, tools: list) -> str:
    """
    渲染工具调用生成提示

    参数:
        thought: 思考过程字符串
        tools: 工具定义列表

    返回:
        渲染后的提示字符串
    """
    template = get_tool_call_gen_template()

    # 转换工具对象为字典格式
    tools_dict = []
    for tool in tools:
        tool_dict = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters.model_dump()
            if hasattr(tool.parameters, "model_dump")
            else tool.parameters,
        }
        tools_dict.append(tool_dict)

    return template.render(thought=thought, tools=tools_dict)


def get_assistant_reply_template():
    """
    获取助手回复模板

    返回:
        编译后的Jinja2模板对象
    """
    return _load_template("assistant_reply")


def render_assistant_reply_prompt(thought: str, conversation_history: list) -> str:
    """
    渲染助手回复提示

    参数:
        thought: 思考过程字符串
        conversation_history: 对话历史消息列表

    返回:
        渲染后的提示字符串
    """
    template = get_assistant_reply_template()

    # 转换消息对象为字典格式
    history_dict = []
    for message in conversation_history:
        msg_dict = {"role": message.role, "content": message.content}
        history_dict.append(msg_dict)

    return template.render(thought=thought, conversation_history=history_dict)
