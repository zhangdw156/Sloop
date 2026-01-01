import datetime
import os
import sys

from evalscope import TaskConfig, run_task

# 定义 BFCL v3 的完整子集列表
BFCL_V3_FULL_SUBSETS = [
    "simple",
    "parallel",
    "multiple",
    "parallel_multiple",
    "java",
    "javascript",
    "miss_func",
    "chatable",
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
    "long_context",
]


def get_done_subsets(checkpoint_path):
    """读取已完成的子集列表"""
    if not os.path.exists(checkpoint_path):
        return set()
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return set(lines)


def append_to_checkpoint(checkpoint_path, subset_name):
    """记录已完成的子集"""
    try:
        with open(checkpoint_path, "a", encoding="utf-8") as f:
            f.write(f"{subset_name}\n")
    except Exception as e:
        print(f"⚠️ Warning: Failed to update checkpoint: {e}")


def format_friendly_result(subset_name, raw_result):
    """
    解析复杂的 Report 对象，生成人类可读的成绩单
    """
    try:
        # 1. 获取核心 Report 对象
        report = raw_result.get("bfcl_v3")
        if not report:
            return str(raw_result)

        # 2. 准备输出缓冲区
        lines = []
        lines.append(f"📊 Model:   {getattr(report, 'model_name', 'Unknown')}")

        # 3. 深入挖掘 metrics -> categories -> subsets 找到分数
        found_data = False
        if hasattr(report, "metrics"):
            for metric in report.metrics:
                if hasattr(metric, "categories"):
                    for cat in metric.categories:
                        if hasattr(cat, "subsets"):
                            for sub in cat.subsets:
                                # 只提取当前正在跑的这个子集的分数
                                if sub.name == subset_name:
                                    lines.append(f"🎯 Subset:  {sub.name}")
                                    lines.append(f"🔢 Samples: {sub.num}")
                                    # 将分数转换为百分比显示，保留2位小数
                                    score_pct = (
                                        sub.score * 100
                                        if sub.score <= 1.0
                                        else sub.score
                                    )
                                    lines.append(
                                        f"🏆 Score:   {sub.score:.4f} ({score_pct:.2f}%)"
                                    )
                                    found_data = True

        if found_data:
            return "\n".join(lines)
        else:
            return str(report)

    except Exception as e:
        return f"Error formatting result: {e}\nRaw: {str(raw_result)}"


def append_result_text(filepath, subset_name, raw_result):
    """将结果追加到文本文件"""
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 调用格式化函数
        formatted_content = format_friendly_result(subset_name, raw_result)

        with open(filepath, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 25} {timestamp} {'=' * 25}\n")
            f.write(formatted_content)
            f.write(f"\n{'=' * 72}\n\n")

        print(f"💾 Result saved to {filepath}")
    except Exception as e:
        print(f"❌ Error saving text result: {e}")


def main():
    # --- 1. 读取基础环境变量 ---
    model_name = os.getenv("EVAL_MODEL_NAME")
    api_url = os.getenv("EVAL_API_URL")
    api_key = os.getenv("EVAL_API_KEY", "EMPTY")
    output_dir = os.getenv("EVAL_OUTPUT_DIR")
    max_tokens = int(os.getenv("EVAL_MAX_TOKENS", "32000"))

    limit_env = os.getenv("EVAL_LIMIT")
    eval_limit = int(limit_env) if limit_env and int(limit_env) > 0 else None

    # --- 2. 确定要跑的子集 ---
    subset_env = os.getenv("EVAL_SUBSET_LIST", "")
    if subset_env.strip():
        target_subsets = [s.strip() for s in subset_env.split(",")]
    else:
        print("ℹ️ No subset list provided. Using FULL BFCL v3 list.")
        target_subsets = BFCL_V3_FULL_SUBSETS

    print(f"🔧 Config: Model={model_name}")
    print(f"📂 Output Dir: {output_dir}")

    # --- 3. 初始化文件路径 ---
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_txt_path = os.path.join(output_dir, "evaluation_results.txt")
        checkpoint_path = os.path.join(output_dir, "done_subsets.txt")
        done_subsets = get_done_subsets(checkpoint_path)
    else:
        print("❌ Error: EVAL_OUTPUT_DIR is not set.")
        sys.exit(1)

    # --- 4. 循环执行每个子集 ---
    print(f"🚀 Starting execution loop for {len(target_subsets)} subsets...")

    for i, subset in enumerate(target_subsets):
        print(f"\n[{i + 1}/{len(target_subsets)}] Checking subset: {subset}")

        # [断点续传检查]
        if subset in done_subsets:
            print(f"⏩ Subset [{subset}] already in {checkpoint_path}. Skipping.")
            continue

        try:
            # 配置任务
            task_cfg = TaskConfig(
                model=model_name,
                api_url=api_url,
                api_key=api_key,
                eval_type="openai_api",
                datasets=["bfcl_v3"],
                # 指定 EvalScope 的工作目录
                # 这样日志和临时文件会生成在 output_dir 下，而不是默认的 ./outputs
                work_dir=output_dir,
                eval_batch_size=int(os.getenv("EVAL_BATCH_SIZE", "10")),
                dataset_args={
                    "bfcl_v3": {
                        "subset_list": [subset],
                        "extra_params": {
                            "underscore_to_dot": True,
                            "is_fc_model": True,
                        },
                    }
                },
                generation_config={
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "parallel_tool_calls": True,
                },
                limit=eval_limit,
            )

            print(f"▶️ Running subset: {subset} ...")

            # 执行评测
            raw_result = run_task(task_cfg=task_cfg)

            # --- 5. 保存结果 (使用优化后的格式) ---
            append_result_text(result_txt_path, subset, raw_result)

            # --- 6. 更新进度 ---
            append_to_checkpoint(checkpoint_path, subset)
            done_subsets.add(subset)

        except Exception as e:
            err_msg = f"❌ Error running subset [{subset}]: {e}"
            print(err_msg)
            append_result_text(result_txt_path, subset, err_msg)
            continue

    print("\n✅ All Subsets Processed.")


if __name__ == "__main__":
    main()
