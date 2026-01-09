import json
import os
import sys
import asyncio
from pathlib import Path
from typing import List, Dict

# 添加项目根目录到 sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from sloop.agents import UserProxyAgent, AssistantAgent, SimulatorAgent
from sloop.models import UserIntent, TaskSkeleton
from sloop.utils.logger import logger, setup_logging

# ==============================================================================
# 1. 数据加载辅助函数
# ==============================================================================

def load_json_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_tools_from_jsonl(path: str) -> Dict[str, dict]:
    """加载工具定义，返回 {tool_name: tool_dict}"""
    tools_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            tool = json.loads(line)
            # 兼容不同格式，确保拿到 function define
            func_def = tool.get("function", tool)
            tools_map[func_def["name"]] = func_def
    return tools_map

import hashlib

def load_skeletons_map(path: str) -> Dict[str, TaskSkeleton]:
    """加载 Skeleton 并建立 id -> Object 的映射"""
    data = load_json_file(path)
    skel_map = {}
    for item in data:
        try:
            skel = TaskSkeleton(**item)
            # 假设 Intent 里的 skeleton_id 对应的就是这里的 ID 逻辑
            # 如果你的 skeleton 数据里有 id 字段，请直接用 item['id']
            # 这里使用稳定的 md5 哈希替代 Python 的 hash()
            sig = skel.get_edges_signature()
            skel_id = f"skel_{hashlib.md5(sig.encode()).hexdigest()}" 
            
            # 如果 json 里有显式的 id 字段，优先使用
            if "id" in item:
                skel_id = item["id"]
                
            skel_map[skel_id] = skel
        except Exception as e:
            logger.warning(f"Failed to parse skeleton: {e}")
    return skel_map

# ==============================================================================
# 2. 轨迹格式化 (Data Converter)
# ==============================================================================

def format_trajectory(tools_list: List[dict], history: List) -> dict:
    """将 AgentScope 的历史转换为目标训练格式"""
    messages = []
    
    for msg in history:
        role = msg.role
        content = msg.content
        tool_calls = getattr(msg, "tool_calls", None)
        
        if role == "user":
            if content in ["TERMINATE", "TERMINATE_FAILED"]: continue
            messages.append({"role": "user", "content": content})
            
        elif role == "assistant":
            if tool_calls:
                for tc in tool_calls:
                    # 兼容对象或字典
                    func = tc.function if hasattr(tc, 'function') else tc.get('function')
                    # OpenAI 格式要求 arguments 是 string
                    args_str = func.arguments if hasattr(func, 'arguments') else func.get('arguments')
                    
                    # 为了存入 dataset，我们通常希望它是 parsed object
                    args_obj = {}
                    if isinstance(args_str, str):
                        try:
                            args_obj = json.loads(args_str)
                        except: pass
                    elif isinstance(args_str, dict):
                        args_obj = args_str

                    t_call = {
                        "name": func.name if hasattr(func, 'name') else func.get('name'),
                        "arguments": args_obj
                    }
                    messages.append({
                        "role": "tool_call",
                        "content": json.dumps(t_call, ensure_ascii=False)
                    })
            else:
                messages.append({"role": "assistant", "content": content})
        
        elif role == "tool":
            messages.append({"role": "tool_response", "content": content})

    return {
        "tools": json.dumps(tools_list, ensure_ascii=False),
        "messages": messages
    }

# ==============================================================================
# 3. 主流程 (异步)
# ==============================================================================

async def run_simulation_loop():
    setup_logging()
    
    # --- 配置路径 ---
    intent_path = project_root / "data/intents/intents_neighborhood.json"
    skeleton_path = project_root / "data/samples/skeletons_neighborhood.json"
    tools_path = project_root / "data/tools.jsonl"

    
    # --- 2. 加载数据 ---
    logger.info("Loading data resources...")
    full_tool_map = load_tools_from_jsonl(tools_path)
    skeleton_map = load_skeletons_map(skeleton_path)
    intents_data = load_json_file(intent_path)
    
    logger.info(f"Loaded {len(full_tool_map)} tools, {len(skeleton_map)} skeletons, {len(intents_data)} intents.")

    # --- 3. 运行验证 ---
    for i, intent_dict in enumerate(intents_data[:2]):
        logger.info(f"\n{'='*20} Running Simulation {i+1} {'='*20}")
        
        # A. 准备对象
        try:
            intent = UserIntent(**intent_dict)
        except Exception as e:
            logger.error(f"Intent parsing failed: {e}")
            continue

        # 关联 Skeleton
        skel_id = intent.meta.get("skeleton_id")
        skeleton = skeleton_map.get(skel_id)
        
        if not skeleton:
            logger.warning(f"Skeleton {skel_id} not found! Skipping.")
            continue
        
        # 准备 Assistant 需要的工具列表
        available_tool_names = intent.available_tools
        task_tools = [full_tool_map.get(name) for name in available_tool_names if full_tool_map.get(name)]

        # B. 实例化 Agents
        # [修改] 移除 model_config_name 参数
        user_agent = UserProxyAgent(name="User", intent=intent)
        
        # AssistantAgent 不需要 config name 了
        assist_agent = AssistantAgent(
            name="Assistant", 
            tools_list=task_tools
        )
        
        sim_agent = SimulatorAgent(name="Environment", intent=intent, skeleton=skeleton)

        # C. 编排循环
        history = []
        user_msg = None 
        
        logger.info(f"Query: {intent.query}")

        while True:
            # 1. User Speaks / Checks
            user_msg = await user_agent.reply(user_msg)
            history.append(user_msg)
            
            if user_msg.content in ["TERMINATE", "TERMINATE_FAILED"]:
                break
            
            # 2. Assistant Thinks
            assist_msg = await assist_agent.reply(user_msg)
            history.append(assist_msg)
            
            # 3. Handle Tool Calls Loop
            while getattr(assist_msg, "tool_calls", None):
                # Simulator Execute
                tool_outputs = await sim_agent.reply(assist_msg)
                
                # 记录 Tool Output
                if isinstance(tool_outputs, list):
                    history.extend(tool_outputs)
                    tool_feed_back = tool_outputs
                else:
                    history.append(tool_outputs)
                    tool_feed_back = tool_outputs
                
                # Assistant Observes Tool Output & Thinks Again
                assist_msg = await assist_agent.reply(tool_feed_back)
                history.append(assist_msg)
            
            # 4. 传递给下一轮 User
            user_msg = assist_msg

        # D. 格式化并保存结果
        final_data = format_trajectory(task_tools, history)
        
        # 确保输出目录存在
        output_dir = project_root / "data" / "verify_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存到文件，每个模拟一个文件
        output_file = output_dir / f"simulation_{i+1}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n--- Generated Training Data Sample saved to {output_file} ---")
        print(json.dumps(final_data, indent=2, ensure_ascii=False))

def main():
    asyncio.run(run_simulation_loop())

if __name__ == "__main__":
    main()