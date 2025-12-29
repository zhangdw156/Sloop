"""
Sloop CLI 入口

使用 typer 实现命令行接口，用于生成多轮工具调用对话数据。
"""

import json
import logging
from pathlib import Path
from typing import Optional, List

import typer
from tqdm import tqdm

from ..engine import BlueprintGenerator
from ..engine.fsm import ConversationLoop
from ..models import ToolDefinition, ChatMessage, ToolCall

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer()


@app.callback()
def main():
    """
    Sloop - 多轮工具调用数据生成框架
    """
    pass


def convert_to_training_format(tools: List[ToolDefinition], messages: List[ChatMessage]) -> dict:
    """
    将内部消息格式转换为训练数据格式

    参数:
        tools: 活跃的工具定义列表
        messages: 对话消息列表

    返回:
        训练数据格式的字典
    """
    # 转换tools为JSON字符串
    tools_list = [tool.model_dump() for tool in tools]
    tools_str = json.dumps(tools_list, ensure_ascii=False)

    # 转换messages
    converted_messages = []
    for msg in messages:
        if msg.role == "user":
            # 用户消息保持不变
            converted_messages.append({
                "role": "user",
                "content": msg.content
            })
        elif msg.role == "assistant" and msg.tool_call:
            # 助手消息（有工具调用）-> tool_call
            tool_call_data = {
                "name": msg.tool_call.name,
                "arguments": msg.tool_call.arguments
            }
            converted_messages.append({
                "role": "tool_call",
                "content": json.dumps(tool_call_data, ensure_ascii=False)
            })
        elif msg.role == "tool":
            # 工具响应 -> tool_response
            converted_messages.append({
                "role": "tool_response",
                "content": msg.content
            })
        elif msg.role == "assistant":
            # 助手消息（无工具调用）保持不变
            converted_messages.append({
                "role": "assistant",
                "content": msg.content
            })

    return {
        "tools": tools_str,
        "messages": converted_messages
    }


@app.command()
def generate(
    input_file: str = typer.Option("tests/data/tools.json", "--input", "-i", help="工具定义文件路径"),
    output_file: str = typer.Option("output.jsonl", "--output", "-o", help="输出文件路径"),
    count: int = typer.Option(1, "--count", "-c", help="生成对话数量"),
    max_turns: int = typer.Option(20, "--max-turns", "-t", help="最大对话轮数"),
    chain_length: int = typer.Option(3, "--chain-length", "-l", help="工具链长度"),
):
    """
    生成多轮工具调用对话数据

    从工具定义文件中读取工具，自动生成对话蓝图和完整的对话流程。
    """
    typer.echo(f"🚀 开始生成 {count} 个对话数据")
    typer.echo(f"   📥 输入文件: {input_file}")
    typer.echo(f"   📤 输出文件: {output_file}")
    typer.echo(f"   🔄 最大轮数: {max_turns}")
    typer.echo(f"   🔗 工具链长度: {chain_length}")

    # 1. 加载工具定义
    typer.echo("📋 加载工具定义...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            tools_data = json.load(f)

        # 转换为 ToolDefinition 对象
        tools = [ToolDefinition(**tool) for tool in tools_data]
        typer.echo(f"   ✅ 加载了 {len(tools)} 个工具定义")

    except FileNotFoundError:
        typer.echo(f"❌ 找不到输入文件: {input_file}", err=True)
        raise typer.Exit(1)
    except json.JSONDecodeError as e:
        typer.echo(f"❌ JSON解析错误: {e}", err=True)
        raise typer.Exit(1)

    # 2. 初始化蓝图生成器
    typer.echo("🔧 初始化蓝图生成器...")
    generator = BlueprintGenerator(tools)

    # 3. 准备输出文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 4. 生成对话数据
    typer.echo("🎬 开始生成对话...")

    with tqdm(total=count, desc="生成进度") as pbar:
        for i in range(count):
            try:
                # 生成蓝图
                blueprint = generator.generate(chain_length=chain_length)

                # 根据blueprint.required_tools筛选active_tools
                active_tools = [
                    tool for tool in tools
                    if tool.name in blueprint.required_tools
                ]
                typer.echo(f"   🔧 使用 {len(active_tools)} 个活跃工具: {blueprint.required_tools}")

                # 创建对话循环（只传入active_tools，防止Context溢出）
                conversation_id = f"conv_{i+1:04d}"
                loop = ConversationLoop(blueprint, active_tools, conversation_id, max_turns=max_turns)

                # 运行对话
                loop.run()

                # 转换为训练数据格式
                training_data = convert_to_training_format(active_tools, loop.context.messages)

                # 追加写入输出文件
                with open(output_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(training_data, ensure_ascii=False) + '\n')

                pbar.set_description(f"生成进度 (最近: {blueprint.intent[:20]}...)")

            except Exception as e:
                logger.error(f"生成对话 {i+1} 失败: {e}")
                typer.echo(f"⚠️ 跳过失败的对话 {i+1}: {e}", err=True)
                continue

            pbar.update(1)

    typer.echo(f"✅ 生成完成！输出文件: {output_file}")


if __name__ == "__main__":
    app()
