import json


def convert_apigen_to_swift_standard(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []

    for entry in data:
        # 1. 构建 messages 列表
        new_messages = []
        for msg in entry["conversations"]:
            raw_role = msg["from"]
            content = msg["value"]

            # --- 核心角色映射逻辑 ---
            if raw_role == "human":
                role = "user"
            elif raw_role == "gpt":
                role = "assistant"
            elif raw_role == "function_call":
                # APIGen 的 function_call 内容通常是 JSON 格式的参数
                # Swift 的 tool_call 需要是一个包含 name 和 arguments 的 JSON 字符串
                # 假设 APIGen 已经是合法的 JSON 字符串，直接透传即可
                role = "tool_call"
            elif raw_role == "observation":
                role = "tool_response"
            else:
                role = "user"  # Fallback

            new_messages.append({"role": role, "content": content})

        # 2. 构建最外层对象
        new_entry = {
            # Swift 支持 tools 字段直接作为对象列表，不一定要转义成字符串
            # 但如果你想严格对齐示例，可以 json.dumps(entry["tools"])
            "tools": entry.get("tools", []),
            "messages": new_messages,
        }

        new_data.append(new_entry)

    # 保存
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 转换成功: {len(new_data)} 条数据已保存至 {output_file}")
    print("格式已适配 ms-swift 标准 tool_call/tool_response 角色。")


# 执行转换
if __name__ == "__main__":
    # 请替换为你实际的文件路径
    convert_apigen_to_swift_standard(
        "/dfs/data/datasets/APIGen-MT-5k/apigen-mt_5k.json",
        "/dfs/data/datasets/APIGen-MT-5k/apigen_swift_standard.json",
    )
