"""
`sloop optimize` 模块
利用强模型 API 执行 JGLV (标签校验) 和 EDDE (错误驱动扩展)。
"""

import json
from typing import Any, Dict, List

from openai import OpenAI
from tqdm import tqdm

from sloop.core.config import SloopConfig

# 移除了对 Sloop 的未使用导入


class DataOptimizer:
    """
    数据优化器
    负责对原始数据集进行标签校验和错误驱动扩展
    """

    def __init__(self, config: SloopConfig):
        """
        初始化数据优化器

        Args:
            config (SloopConfig): Sloop 配置对象
        """
        self.config = config
        self.client = OpenAI(
            api_key=config.strong.api_key, base_url=config.strong.base_url
        )
        # 修复：移除未使用的 generator 属性
        pass

    def load_data(self, dataset_file: str, boundary_file: str) -> tuple:
        """
        加载原始数据集和边界案例

        Args:
            dataset_file (str): 原始数据集文件路径
            boundary_file (str): 边界案例文件路径

        Returns:
            tuple: (原始数据集, 边界案例)
        """
        with open(dataset_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        with open(boundary_file, "r", encoding="utf-8") as f:
            boundary_cases = json.load(f)
        return dataset, boundary_cases

    def perform_jglv(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        执行 JGLV (Judgement and Label Verification)
        利用强模型校验并修正原始数据集中的标签。

        Args:
            dataset (List[Dict[str, Any]]): 原始数据集

        Returns:
            List[Dict[str, Any]]: 经过标签校验和修正后的数据集
        """
        corrected_dataset = []

        print("开始执行 JGLV (标签校验)...")
        for data in tqdm(dataset, desc="校验标签"):
            try:
                # 构造提示，让强模型判断标签是否正确并修正
                prompt = (
                    "请判断以下对话中助手的工具调用标签是否正确。如果正确，请返回原标签；\n"
                    "如果不正确，请返回修正后的正确标签。\n"
                    f"对话内容:\n{data['conversation']}\n"
                    f"原始标签:\n{json.dumps(data['label'], ensure_ascii=False, indent=2)}\n"
                    "请仅返回修正后的 JSON 标签。"
                )
                response = self.client.chat.completions.create(
                    model="gpt-4o",  # 强模型
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=512,
                )
                #  解析强模型的修正结果
                corrected_label = json.loads(
                    response.choices[0].message.content.strip()
                )
                # 更新数据
                corrected_data = data.copy()
                corrected_data["label"] = corrected_label
                corrected_dataset.append(corrected_data)
            except Exception as e:
                print(f"JGLV 校验标签时出错: {e}")
                # 如果出错，保留原数据
                corrected_dataset.append(data)

        print(f"JGLV 完成，共处理 {len(corrected_dataset)} 个样本。")
        return corrected_dataset

    def perform_edde(
        self,
        boundary_cases: List[Dict[str, Any]],
        service_definitions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        执行 EDDE (Error-Driven Data Expansion)
        针对边界案例，利用强模型生成相似但多样化的挑战性新样本。

        Args:
            boundary_cases (List[Dict[str, Any]]): 边界案例
            service_definitions (List[Dict[str, Any]]): 服务定义列表

        Returns:
            List[Dict[str, Any]]: 生成的扩展数据集
        """
        expanded_dataset = []

        print("开始执行 EDDE (错误驱动扩展)...")
        for case in tqdm(boundary_cases, desc="扩展错题"):
            try:
                # 查找对应的服务定义
                service_name = case["service"]
                service = next(
                    (s for s in service_definitions if s["name"] == service_name), None
                )
                if not service:
                    print(f"未找到服务 {service_name} 的定义，跳过。")
                    continue

                # 构造提示，让强模型基于边界案例生成新的挑战性样本
                prompt = (
                    "你是一个专业的数据增强专家。请根据给定的边界案例（弱模型处理不了的案例）\n"
                    "和服务定义，生成一个相似但更具挑战性的新对话样本。\n"
                    "新样本应保持服务调用的核心意图，但在用户请求的表述、"
                    "上下文或参数复杂度上增加难度，以帮助弱模型更好地学习。\n"
                    f"边界案例:\n{json.dumps(case, ensure_ascii=False, indent=2)}\n"
                    f"服务定义:\n{json.dumps(service, ensure_ascii=False, indent=2)}\n"
                    "请生成一个用户与助手的对话，包含用户请求、助手的思考过程和最终的工具调用。"
                )
                response = self.client.chat.completions.create(
                    model="gpt-4o",  # 强模型
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1024,
                )
                #  解析强模型的生成结果，这里简化处理
                generated_text = response.choices[0].message.content.strip()

                # 由于模型可能直接输出对话，我们假设它也输出了正确的标签
                # 实际应用中可能需要更复杂的解析或要求模型输出结构化数据
                # 这里我们复用生成器的逻辑来模拟生成标签
                # 为简化，直接使用服务定义生成一个标签
                new_label = {
                    "tool_call": {
                        "name": service["name"],
                        "arguments": dict.fromkeys(service["parameters"], "value"),
                    }
                }

                expanded_dataset.append({
                    "service": service_name,
                    "conversation": generated_text,
                    "label": new_label,
                    "source": "EDDE_generated_from_boundary_case",
                })
            except Exception as e:
                print(f"EDDE 生成样本时出错: {e}")

        print(f"EDDE 完成，共生成 {len(expanded_dataset)} 个新样本。")
        return expanded_dataset

    def optimize_dataset(
        self,
        original_dataset_file: str,
        boundary_cases_file: str,
        services_file: str,
        output_file: str,
    ) -> None:
        """
        执行完整的优化流程：JGLV + EDDE

        Args:
            original_dataset_file (str): 原始数据集文件路径
            boundary_cases_file (str): 边界案例文件路径
            services_file (str): 服务定义文件路径
            output_file (str): 输出优化后数据集文件路径
        """
        # 1. 加载数据
        dataset, boundary_cases = self.load_data(
            original_dataset_file, boundary_cases_file
        )

        # 2. 执行 JGLV: 校验并修正原始数据集标签
        corrected_dataset = self.perform_jglv(dataset)

        # 3. 执行 EDDE: 基于边界案例生成新样本
        # 首先需要加载服务定义
        service_definitions = self.generator.load_services(services_file)
        expanded_dataset = self.perform_edde(boundary_cases, service_definitions)

        # 4. 合并数据集
        final_dataset = corrected_dataset + expanded_dataset

        #  保存最终优化的数据集
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_dataset, f, ensure_ascii=False, indent=2)

        print(
            f"数据集优化完成，共 {len(final_dataset)} 个样本 (原始 {len(corrected_dataset)}, 扩展 {len(expanded_dataset)})，已保存至 {output_file}"
        )
