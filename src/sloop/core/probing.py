"""
`sloop probe` 模块
利用弱模型 API 执行 Greedy Capability Probing (GCP)。
"""

import json
from typing import List, Dict, Any
from openai import OpenAI
from tqdm import tqdm
from .config import SloopConfig


class CapabilityProber:
    """
    能力探测器
    负责使用弱模型对数据集进行探测，识别边界案例
    """
    def __init__(self, config: SloopConfig):
        """
        初始化能力探测器
        
        Args:
            config (SloopConfig): Sloop 配置对象
        """
        self.config = config
        self.client = OpenAI(
            api_key=config.weak.api_key,
            base_url=config.weak.base_url
        )

    def load_dataset(self, filepath: str) -> List[Dict[str, Any]]:
        """
        从 JSON 文件加载数据集
        
        Args:
            filepath (str): 数据集文件路径
            
        Returns:
            List[Dict[str, Any]]: 数据集
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def probe_single_conversation(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        探测单个对话，检查弱模型的输出与标签是否一致
        
        Args:
            conversation_data (Dict[str, Any]): 单个对话数据
            
        Returns:
            Dict[str, Any]: 探测结果，包含是否为边界案例
        """
        # 提取标签中的工具调用
        expected_tool_call = conversation_data["label"]["tool_call"]
        
        # 构造系统提示，要求模型进行工具调用
        system_prompt = "你是一个助手，需要根据用户请求调用相应的工具。请严格按照 JSON 格式输出工具调用。"
        
        # 使用对话中的用户请求
        user_message = conversation_data["conversation"]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # 假设弱模型为 gpt-3.5-turbo，实际应由配置决定
                model="gpt-3.5-turbo",  # 假设弱模型为 gpt-3.5-turbo，实际应由配置决定
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0,  # Greedy 解码
                max_tokens=512
            )
            
            # 解析弱模型的输出，这里简化处理
            model_output = response.choices[0].message.content.strip()
            
            # 简单比较，实际应用中需要更复杂的逻辑来解析和比较工具调用
            # 例如解析 JSON、比较参数等
            is_correct = self._compare_tool_calls(model_output, expected_tool_call)
            
        except Exception as e:
            #  如果模型调用失败，视为处理不了的案例
            print(f"探测对话时出错: {e}")
            is_correct = False
            model_output = str(e) # 确保 model_output 在 except 块中被定义
        
        return {
            "conversation_id": conversation_data.get("id", "unknown"),
            "service": conversation_data["service"],
            "is_boundary": not is_correct, # 如果不正确，则为边界案例
            "model_output": model_output,  # 现在 model_output 总是被定义
            "model_output": model_output,  # 现在 model_output 总是被定义
            "expected": expected_tool_call
        }

    def _compare_tool_calls(self, model_output: str, expected: Dict[str, Any]) -> bool:
        """
        比较模型输出的工具调用与预期是否一致
        这是一个简化的实现，实际应用中需要更健壮的解析和比较逻辑。
        
        Args:
            model_output (str): 模型输出的字符串
            expected (Dict[str, Any]): 预期的工具调用
            
        Returns:
            bool: 是否一致
        """
        #  简单的字符串包含检查，实际应解析 JSON 并比较结构
        #  这只是一个占位符，实际实现需要更健壮的逻辑
        #  返回是否包含服务名称
        return expected["name"] in model_output
        #  返回是否包含服务名称
        return expected["name"] in model_output

    def run_probe(self, dataset_file: str, output_file: str) -> None:
        """
        执行完整的探测流程
        
        Args:
            dataset_file (str): 输入数据集文件路径
            output_file (str): 输出边界案例文件路径
        """
        dataset = self.load_dataset(dataset_file)
        boundary_cases = []
        
        print(f"开始探测数据集，共 {len(dataset)} 个对话...")
        for data in tqdm(dataset, desc="探测能力"):
            result = self.probe_single_conversation(data)
            if result["is_boundary"]:
                boundary_cases.append(result)
        
        # 保存边界案例
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(boundary_cases, f, ensure_ascii=False, indent=2)
        
        print(f"探测完成，共发现 {len(boundary_cases)} 个边界案例，已保存至 {output_file}")
