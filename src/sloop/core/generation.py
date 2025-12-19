"""
`sloop gen` 模块
利用强模型 API 生成高质量的服务调用对话数据集。
"""

import json
from typing import List, Dict, Any
from openai import OpenAI
from tqdm import tqdm
from .config import SloopConfig


class DataGenerator:
    """
    数据生成器
    负责读取服务定义并使用强模型生成对话数据
    """

    def __init__(self, config: SloopConfig):
        """
        初始化数据生成器

        Args:
            config (SloopConfig): Sloop 配置对象
        """
        self.config = config
        self.client = OpenAI(
            api_key=config.strong.api_key, base_url=config.strong.base_url
        )

    def load_services(self, filepath: str) -> List[Dict[str, Any]]:
        """
        从 JSON 文件加载服务定义

        Args:
            filepath (str): 服务定义文件路径

        Returns:
            List[Dict[str, Any]]: 服务定义列表
        """
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def generate_conversation(self, service: Dict[str, Any]) -> Dict[str, Any]:
        """
        为单个服务生成一个模拟的对话

        Args:
            service (Dict[str, Any]): 服务定义

        Returns:
            Dict[str, Any]: 生成的对话数据
        """
        # 构造系统提示
        system_prompt = """你是一个专业的服务调用模拟器。请根据给定的服务定义，生成一个用户与助手之间的高质量对话。
对话应包含用户请求、助手的思考过程和最终的工具调用。
"""

        # 构造用户消息
        user_prompt = f"""
服务名称: {service["name"]}
服务描述: {service["description"]}
服务参数: {json.dumps(service["parameters"], ensure_ascii=False, indent=2)}

请生成一个用户请求该服务的对话示例。
"""

        response = self.client.chat.completions.create(
            model="gpt-4o",  # 假设强模型为 gpt-4o，实际应由配置决定
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=1024,
        )

        #  解析模型输出，这里简化处理，实际可能需要更复杂的解析
        #  这里假设输出是纯文本，实际应用中可能需要解析JSON或特定格式
        generated_text = response.choices[0].message.content.strip()

        # 构建并返回完整的对话数据
        return {
            "service": service["name"],
            "conversation": generated_text,
            "label": {  # 模拟生成的标签，实际应由模型结构化输出
                "tool_call": {
                    "name": service["name"],
                    "arguments": {param: "value" for param in service["parameters"]},
                }
            },
        }

    def generate_dataset(self, services_file: str, output_file: str) -> None:
        """
        生成完整的数据集

        Args:
            services_file (str): 服务定义文件路径
            output_file (str): 输出数据集文件路径
        """
        services = self.load_services(services_file)
        dataset = []

        print(f"开始生成数据集，共 {len(services)} 个服务...")
        for service in tqdm(services, desc="生成对话"):
            try:
                conversation = self.generate_conversation(service)
                dataset.append(conversation)
            except Exception as e:
                print(f"生成服务 {service['name']} 的对话时出错: {e}")

        # 保存数据集
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        print(f"数据集已生成并保存至 {output_file}")
