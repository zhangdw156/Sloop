import os
import json
import sys
from evalscope import TaskConfig, run_task

def main():
    # --- 1. 读取基础环境变量 ---
    model_name = os.getenv('EVAL_MODEL_NAME')
    api_url = os.getenv('EVAL_API_URL')
    api_key = os.getenv('EVAL_API_KEY', 'EMPTY')
    output_dir = os.getenv('EVAL_OUTPUT_DIR')
    max_tokens = int(os.getenv('EVAL_MAX_TOKENS', '32000'))
    
    # Limit设置
    limit_env = os.getenv('EVAL_LIMIT')
    eval_limit = int(limit_env) if limit_env and int(limit_env) > 0 else None

    # --- [新增] 读取子集列表环境变量 ---
    # 期望格式: "multi_turn_base,multi_turn_miss" (逗号分隔)
    subset_env = os.getenv('EVAL_SUBSET_LIST', '')
    # 如果环境变量存在且不为空，则分割成列表；否则为 None (跑全量)
    target_subsets = [s.strip() for s in subset_env.split(',')] if subset_env.strip() else None

    print(f"🔧 Config: Model={model_name}")
    print(f"🎯 Target Subsets: {target_subsets if target_subsets else 'ALL'}")

    # --- 配置任务 ---
    task_cfg = TaskConfig(
        model=model_name,
        api_url=api_url,
        api_key=api_key,
        eval_type='openai_api',
        datasets=['bfcl_v3'],
        eval_batch_size=int(os.getenv('EVAL_BATCH_SIZE', '10')),
        dataset_args={
            'bfcl_v3': {
                # [关键修改] 将子集列表传给 EvalScope
                'subset_list': target_subsets, 
                'extra_params': {
                    'underscore_to_dot': True,
                    'is_fc_model': True,
                }
            }
        },
        generation_config={
            'temperature': 0,
            'max_tokens': max_tokens, 
            'parallel_tool_calls': True,
        },
        limit=eval_limit, 
    )

    # --- 执行评测 ---
    try:
        result = run_task(task_cfg=task_cfg)
        
        if output_dir:
            res_path = os.path.join(output_dir, "result_summary.json")
            with open(res_path, "w") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            print(f"✅ Python Script Finished. Summary saved to {res_path}")

    except Exception as e:
        print(f"❌ Python Execution Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()