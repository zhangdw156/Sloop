# -*- coding: utf-8 -*-
import json
import os

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# ================= 配置区域 =================
# 模型路径
MODEL_PATH = "/dfs/data/models/Qwen3-0.6B"

# 数据集路径 (根据你截图中的路径推测，如果不对请修改)
DATA_PATH = "/dfs/data/datasets/bfcl_v3/data/train-00000-of-00001.parquet"

# 是否只计算 multi_turn 为 True 的数据
ONLY_MULTI_TURN = True
# ===========================================


def main():
    print(f"🚀 正在加载 Tokenizer: {MODEL_PATH} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"❌ 加载 Tokenizer 失败: {e}")
        return

    print(f"📂 正在读取数据集: {DATA_PATH} ...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ 文件不存在: {DATA_PATH}")
        return

    df = pd.read_parquet(DATA_PATH)

    # 筛选多轮对话
    if ONLY_MULTI_TURN and "multi_turn" in df.columns:
        original_len = len(df)
        df = df[df["multi_turn"]].copy()
        print(f"ℹ️ 已筛选 multi_turn=True 数据: {len(df)} 条 (原数据 {original_len} 条)")
    else:
        print(f"ℹ️ 使用全量数据: {len(df)} 条")

    print("mb 正在计算 Token 数量 (这可能需要几秒钟)...")

    # 定义计算函数
    def get_token_len(row_turns):
        try:
            # 1. 解析数据格式
            # 如果是字符串，先转 JSON
            data = json.loads(row_turns) if isinstance(row_turns, str) else row_turns

            # 处理 BFCL 数据集常见的嵌套列表结构 [[{...}, {...}]]
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                conversation = data[0]
            else:
                conversation = data

            if not isinstance(conversation, list):
                return 0

            # 2. 使用 apply_chat_template 获取最精确的 token 数
            # 这会自动加上 <|im_start|>, <|im_end|>, system prompt 等所有特殊 token
            # tokenize=True 会直接返回 token id 列表
            token_ids = tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=False,  # 训练/评测数据通常不需要生成 prompt
            )
            return len(token_ids)

        except Exception:
            # 如果 apply_chat_template 失败（例如数据格式缺少 role），回退到粗略计算
            # print(f"Warning: Template failed, fallback to raw concat. Error: {e}")
            try:
                full_text = ""
                for turn in conversation:
                    content = turn.get("content")
                    if content:
                        full_text += str(content)
                return len(tokenizer.encode(full_text))
            except Exception:
                return 0

    # 使用 tqdm 显示进度条
    tqdm.pandas(desc="Processing")
    df["token_count"] = df["turns"].progress_apply(get_token_len)

    # 获取结果
    max_tokens = df["token_count"].max()
    max_idx = df["token_count"].idxmax()

    # 获取最长那条数据的详细信息
    longest_row = df.loc[max_idx]

    print("\n" + "=" * 40)
    print("📊 统计结果 (基于 Qwen3 Tokenizer)")
    print("=" * 40)
    print(f"✅ 最长的一条数据包含: {max_tokens} tokens")
    print(f"📍 数据索引 (Index): {max_idx}")
    print(f"📂 所属子集 (subset): {longest_row.get('subset', 'N/A')}")
    print("-" * 40)

    # 检查是否超过 32k
    if max_tokens > 32000:
        print(f"⚠️ 警告: 最大长度 ({max_tokens}) 超过了 32000！")
        print("💡 建议: 在评测脚本中调大 max_model_len 或减少 max_tokens 参数。")
    else:
        print(f"ok 最大长度 ({max_tokens}) 在 32000 安全范围内。")
        print(f"   剩余空间 (32k - max): {32000 - max_tokens}")


if __name__ == "__main__":
    main()
