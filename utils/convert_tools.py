import json
import os

def convert_to_openai_tool_format(input_file="source_tools.json", output_file="openai_tools.jsonl"):
    # 1. 检查输入文件
    if not os.path.exists(input_file):
        print(f"❌ 错误: 找不到输入文件 '{input_file}'")
        # 创建示例文件用于测试
        create_dummy_input(input_file)
        print(f"⚠️  已自动生成测试文件: {input_file}")

    print(f"🔄 正在处理 {input_file} ...")

    try:
        # 读取源文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print("❌ 错误: 源文件必须是一个 JSON 列表 (Array)")
            return

        count = 0
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for item in data:
                # --- 核心转换逻辑 ---
                
                # 1. 清洗 parameters (移除 OpenAI 不支持的字段，如 'optional')
                raw_params = item.get("parameters", {})
                clean_params = {
                    "type": raw_params.get("type", "object"),
                    "properties": raw_params.get("properties", {}),
                    "required": raw_params.get("required", [])
                }
                
                # 2. 组装 Function 结构
                function_body = {
                    "name": item.get("name"),
                    "description": item.get("description", ""),
                    "parameters": clean_params
                }

                # 3. 组装最终的 OpenAI Tool 结构
                # 这就是你想要的每一行的样子：{"type": "function", "function": {...}}
                openai_tool = {
                    "type": "function",
                    "function": function_body
                }

                # 4. 直接写入 JSON 对象 (不二次序列化，不加外层 wrapper)
                f_out.write(json.dumps(openai_tool, ensure_ascii=False) + "\n")
                
                count += 1

        print(f"✅ 成功! 已生成: {output_file}")
        print(f"📊 共转换 {count} 行数据")

    except json.JSONDecodeError:
        print("❌ 错误: 输入文件 JSON 格式不正确")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")

def create_dummy_input(filename):
    """生成测试数据"""
    dummy_data = [
      {
        "name": "racecards_for_greyhound_racing_uk",
        "description": "Get races list...",
        "parameters": {
          "type": "object",
          "properties": {},
          "required": [],
          "optional": []
        },
        "category": "greyhound"
      },
      {
        "name": "example_tool_2",
        "description": "Another tool",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
      }
    ]
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dummy_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    convert_to_openai_tool_format(input_file="/root/work/Sloop/tests/data/tools.json", output_file="/root/work/Sloop/tests/data/tools.jsonl")