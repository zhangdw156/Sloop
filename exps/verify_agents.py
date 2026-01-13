import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

from agentscope.message import Msg

from sloop.agent import AssistantAgent, SimulatorAgent, UserProxyAgent
from sloop.schemas import TaskSkeleton, UserIntent
from sloop.utils import logger, setup_logging

# 添加项目根目录到 sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))
# ==============================================================================
# 1. 数据加载辅助函数
# ==============================================================================


def load_json_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tools_from_jsonl(path: str) -> Dict[str, dict]:
    tools_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            tool = json.loads(line)
            func_def = tool.get("function", tool)
            tools_map[func_def["name"]] = func_def
    return tools_map


def load_skeletons_map(path: str) -> Dict[str, TaskSkeleton]:
    data = load_json_file(path)
    skel_map = {}
    for item in data:
        try:
            skel = TaskSkeleton(**item)
            sig = skel.get_edges_signature()
            skel_id = f"skel_{hashlib.md5(sig.encode()).hexdigest()}"
            if "id" in item:
                skel_id = item["id"]
            skel_map[skel_id] = skel
        except Exception as e:
            logger.warning(f"Failed to parse skeleton: {e}")
    return skel_map


# ==============================================================================
# 2. 核心：将 ReAct Memory 转换为训练数据格式
# ==============================================================================


def format_agent_memory(tools_list: List[dict], memory_msgs: List[Msg]) -> dict:
    """
    将 AssistantAgent (ReAct) 的内存转换为 OpenAI 兼容的 SFT 格式。

    AgentScope Memory 结构映射逻辑:
    - User Msg -> role: "user"
    - Assistant Msg (Text + ToolUseBlock) -> role: "assistant", content=text, tool_calls=[...]
    - System Msg (ToolResultBlock) -> role: "tool", content=output, tool_call_id=...
    """
    messages = []

    for msg in memory_msgs:
        # 1. 处理 System Prompt (通常在 Memory 第一条或由 ReAct 自动注入)
        if msg.role == "system":
            # 检查是否是 Tool Result (AgentScope 将工具结果标记为 system role)
            tool_results = msg.get_content_blocks("tool_result")
            if tool_results:
                for tr in tool_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tr.get("id"),
                        "name": tr.get("name"),
                        "content": str(tr.get("output")),
                    })
            else:
                # 普通 System Prompt
                messages.append({"role": "system", "content": msg.get_text_content()})

        # 2. 处理 User 输入
        elif msg.role == "user":
            content = msg.get_text_content()
            # 过滤掉 UserProxy 产生的终止指令，不放入训练数据
            if content not in ["TERMINATE", "TERMINATE_FAILED"]:
                messages.append({"role": "user", "content": content})

        # 3. 处理 Assistant 输出 (思考 + 工具调用)
        elif msg.role == "assistant":
            openai_msg = {"role": "assistant", "content": None}

            # A. 提取文本内容 (Chain of Thought)
            text_content = msg.get_text_content()
            if text_content:
                openai_msg["content"] = text_content

            # B. 提取工具调用
            tool_uses = msg.get_content_blocks("tool_use")
            if tool_uses:
                tool_calls = []
                for tu in tool_uses:
                    # 将参数对象转回 JSON 字符串，符合 OpenAI 格式
                    args_str = json.dumps(tu.get("input", {}), ensure_ascii=False)
                    tool_calls.append({
                        "id": tu.get("id"),
                        "type": "function",
                        "function": {"name": tu.get("name"), "arguments": args_str},
                    })
                openai_msg["tool_calls"] = tool_calls

            messages.append(openai_msg)

    return {"tools": tools_list, "messages": messages}


# ==============================================================================
# 3. 主流程 (User <-> ReAct Assistant)
# ==============================================================================


async def run_simulation_loop():
    setup_logging()

    # --- 配置路径 ---
    intent_path = project_root / "data/intents/intents_neighborhood.json"
    skeleton_path = project_root / "data/samples/skeletons_neighborhood.json"
    tools_path = project_root / "data/tools.jsonl"
    output_dir = project_root / "data" / "verify_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 加载数据 ---
    logger.info("Loading data resources...")
    full_tool_map = load_tools_from_jsonl(tools_path)
    skeleton_map = load_skeletons_map(skeleton_path)
    intents_data = load_json_file(intent_path)

    # --- 运行循环 ---
    # 限制运行数量方便测试，实际跑可以去掉
    for i, intent_dict in enumerate(intents_data[:5]):
        logger.info(f"\n{'=' * 20} Running Simulation {i + 1} {'=' * 20}")

        # 1. 准备 Context
        try:
            intent = UserIntent(**intent_dict)
        except Exception as e:
            logger.error(f"Intent parsing failed: {e}")
            continue

        skeleton = skeleton_map.get(intent.meta.get("skeleton_id"))
        if not skeleton:
            logger.warning(f"Skeleton not found for Intent {intent.id}")
            continue

        task_tools = [
            full_tool_map.get(n) for n in intent.available_tools if full_tool_map.get(n)
        ]

        # 2. 初始化 Agents
        # Simulator 作为 Environment 存在
        sim_agent = SimulatorAgent(name="Environment", intent=intent, skeleton=skeleton)

        # Assistant (ReAct) 持有 Simulator
        assist_agent = AssistantAgent(
            name="Assistant",
            tools_list=task_tools,
            simulator=sim_agent,  # 注入 Simulator
            max_iters=10,  # ReAct 最大思考步数
            verbose=True,
        )

        # User Proxy
        user_agent = UserProxyAgent(name="User", intent=intent, max_turns=10)

        # 3. 对话循环 (User <-> Assistant)
        # Assistant 的 ReAct 内部循环被封装在 reply 中
        # 这里只看 User 和 Assistant 的交互

        logger.info(f"Query: {intent.query}")

        # 用于传递消息的临时变量
        last_msg = None

        # 只需要简单的回合制，因为 ReAct 会一次性跑完 "思考-调用-结果-思考-回答" 的全过程
        # 直到它决定输出最终文本给 User
        while True:
            # --- User Turn ---
            user_msg = await user_agent.reply(last_msg)

            # 检查终止条件
            if user_msg.get_text_content() in ["TERMINATE", "TERMINATE_FAILED"]:
                logger.info(
                    f"Conversation ended by User: {user_msg.get_text_content()}"
                )
                break

            # --- Assistant Turn (ReAct Loop happens inside) ---
            # Assistant 会执行多步推理，直到产生最后给 User 的回复
            # 中间的工具调用过程都在 Assistant 内部处理并记录在 Memory 中
            assist_msg = await assist_agent.reply(user_msg)

            last_msg = assist_msg

        # 4. 保存数据
        # 直接导出 Assistant 的 Memory，它包含了最完整的视角 (包括 System Prompt, User Query, Thoughts, Tool Calls, Tool Results)
        final_memory = await assist_agent.memory.get_memory()

        formatted_data = format_agent_memory(task_tools, final_memory)

        output_file = output_dir / f"traj_{intent.id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved trajectory to {output_file}")


def main():
    asyncio.run(run_simulation_loop())


if __name__ == "__main__":
    main()
