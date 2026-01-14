from agentscope.message import Msg
from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter
import agentscope
import asyncio
import json
import re
from sloop.configs import env_config

# --- 引入 Rich 库 ---
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.json import JSON
from rich.markdown import Markdown
from rich.theme import Theme

# --- 配置 Rich ---
custom_theme = Theme({
    "user": "green",
    "assistant": "blue",
    "tool": "yellow",
    "sandbox": "magenta",
    "info": "dim white"
})
console = Console(theme=custom_theme, record=True)

# --- 配置模型 ---
MODEL_NAME = env_config.get("OPENAI_MODEL_NAME")
API_KEY = env_config.get("OPENAI_MODEL_API_KEY")
BASE_URL = env_config.get("OPENAI_MODEL_BASE_URL")

# --- 辅助函数：解析消息内容 ---
def get_content_str(msg):
    if hasattr(msg, "content"):
        content = msg.content
    else:
        return str(msg)

    if isinstance(content, list):
        text_content = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_content += item.get("text", "")
        return text_content
    return str(content)

# --- 辅助函数：解析 JSON 工具调用 ---
def parse_tool_calls(content):
    content = str(content).strip()
    # 1. 尝试提取 Markdown 代码块
    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", content)
    if json_match:
        json_str = json_match.group(1)
    else:
        # 2. 尝试查找最外层的 [ ]
        start = content.find('[')
        end = content.rfind(']')
        if start != -1 and end != -1:
            json_str = content[start:end+1]
        else:
            return None

    try:
        if "}{" in json_str: 
            json_str = f"[{json_str.replace('}{', '},{')}]"
        parsed = json.loads(json_str)
        if isinstance(parsed, list): return parsed
        elif isinstance(parsed, dict) and ("tool_call" in parsed or "name" in parsed): return [parsed]
        return parsed
    except json.JSONDecodeError:
        return None

# --- Agent 定义 ---

# User Prompt 增强：防止角色互换
user = ReActAgent(
    name="User",
    sys_prompt="""你是在中国矿业大学上学的大四学生“张三”。

    【绝对指令】
    1. 你是**客户/游客**，你的对话对象是AI助手。
    2. **绝对不要**扮演助手！**绝对不要**帮助对方订票！**绝对不要**列出推荐清单！
    3. 你只需要提出需求、询问价格、提供自己的身份信息。
    4. 只有当助手明确说“预订成功”后，你确认满意了，才能回复 "exit"。

    【当前任务】
    你打算下周五（10月25日）带女朋友“李婷”去上海迪士尼。
    - 询问票价（要两张）。
    - 觉得乐园酒店太贵，询问有没有便宜点的官方酒店（如玩具总动员）。
    - 决定住玩具总动员酒店，并提供身份证信息让助手预订。
    - 最后要求预订一家浪漫的餐厅。
    """,
    model=OpenAIChatModel(model_name=MODEL_NAME, api_key=API_KEY, client_kwargs={"base_url":BASE_URL}),
    formatter=OpenAIChatFormatter() 
)
user.set_console_output_enabled(False)
assistant = ReActAgent(
    name="Assistant",
    sys_prompt="""你是一个专业的迪士尼行程规划AI助手。

    【思考链】
    1. 用户提问 -> 判断是否需要数据 -> 生成 JSON Tool Call。
    2. 获得 Sandbox 数据 -> 结合数据回答用户。

    【输出规则】
    1. 若需调用工具，仅输出 ```json [...] ``` 代码块。
    2. 若不需要工具，用热情专业的口吻回复用户。
    3. 你的工具：search_ticket, search_hotel, book_ticket, book_hotel, reserve_restaurant。
    """,
    model=OpenAIChatModel(model_name=MODEL_NAME, api_key=API_KEY, client_kwargs={"base_url":BASE_URL}),
    formatter=OpenAIChatFormatter() 
)
assistant.set_console_output_enabled(False)

sandbox = ReActAgent(
    name="Sandbox",
    sys_prompt="""你是一个API沙盒。
    1. 接收 JSON 请求。
    2. 返回 JSON 响应。
    3. 模拟真实数据：
       - 10月25日门票 475元。
       - 玩具总动员酒店 1350元（含早）。
       - 皇家宴会厅有位。
    """,
    model=OpenAIChatModel(model_name=MODEL_NAME, api_key=API_KEY, client_kwargs={"base_url":BASE_URL}),
    formatter=OpenAIChatFormatter() 
)
sandbox.set_console_output_enabled(False)

# --- 主逻辑 ---
async def main():
    msg = None
    max_turns = 10
    
    console.print(Panel("🚀 Starting Multi-Agent Simulation: User vs Assistant vs Sandbox", style="bold white on blue"))
    
    for i in range(max_turns):
        turn_num = i + 1
        
        # --- 1. User Turn ---
        msg = await user.reply(msg)
        user_content = get_content_str(msg)
        
        console.print(Panel(
            Markdown(user_content), 
            title=f"[user]Turn {turn_num}: User (张伟)[/user]", 
            border_style="green",
            expand=False
        ))

        if "exit" in user_content.lower():
            console.print("[bold red]>>> User requested exit. Conversation ended.[/bold red]")
            break

        # --- 2. Assistant Turn ---
        msg = await assistant.reply(msg)
        assistant_content = get_content_str(msg)

        # 检测工具调用
        tool_calls = parse_tool_calls(assistant_content)

        if tool_calls:
            # 显示工具调用意图
            console.print(Panel(
                JSON(json.dumps(tool_calls)), 
                title=f"[tool]Turn {turn_num}: Assistant invokes Tools[/tool]", 
                border_style="yellow",
                expand=False
            ))
            
            # --- 3. Sandbox Turn ---
            sandbox_input_msg = Msg(name="System", role="system", content=json.dumps(tool_calls))
            sb_msg = await sandbox.reply(sandbox_input_msg)
            sb_content = get_content_str(sb_msg)
            
            # 尝试格式化 Sandbox 的 JSON 输出
            try:
                sb_display = JSON(sb_content)
            except:
                sb_display = sb_content

            console.print(Panel(
                sb_display, 
                title=f"[sandbox]Turn {turn_num}: Sandbox Return[/sandbox]", 
                border_style="magenta", 
                padding=(0, 2),
                expand=False
            ))
            
            # --- 4. Assistant Final Response (After Tools) ---
            msg = await assistant.reply(sb_msg)
            assistant_final_content = get_content_str(msg)
            
            console.print(Panel(
                Markdown(assistant_final_content), 
                title=f"[assistant]Turn {turn_num}: Assistant (Final Response)[/assistant]", 
                border_style="blue",
                expand=False
            ))
        else:
            # 没有工具调用，直接显示 Assistant 的回复
            console.print(Panel(
                Markdown(assistant_content), 
                title=f"[assistant]Turn {turn_num}: Assistant[/assistant]", 
                border_style="blue",
                expand=False
            ))

if __name__ == "__main__":
    agentscope.init(logging_level="CRITICAL")
    asyncio.run(main())
    console.save_html("data/simulation_report.html")