import os
import json
import sys
from evalscope import TaskConfig, run_task

def main():
    # --- 读取环境变量 ---
    model_name = os.getenv('EVAL_MODEL_NAME')
    api_url = os.getenv('EVAL_API_URL')
    api_key = os.getenv('EVAL_API_KEY', 'EMPTY')
    output_dir = os.getenv('EVAL_OUTPUT_DIR')
    
    # 获取 eval_limit，如果未设置或为 -1 则为 None (跑全量)
    limit_env = os.getenv('EVAL_LIMIT')
    eval_limit = int(limit_env) if limit_env and int(limit_env) > 0 else None

    print(f"🔧 Config: Model={model_name}, URL={api_url}, Limit={eval_limit}")

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
                'extra_params': {
                    'underscore_to_dot': True,
                    'is_fc_model': True,
                }
            }
        },
        generation_config={
            'temperature': 0,
            'max_tokens': 32000,
            'parallel_tool_calls': True,
        },
        limit=eval_limit, 
    )

    # --- 执行评测 ---
    try:
        # run_task 内部会自动打印很多日志
        result = run_task(task_cfg=task_cfg)
        
        # 也可以选择性地把 result 存成 json，虽然 EvalScope 通常自己也会存
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