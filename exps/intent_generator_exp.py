import json
from pathlib import Path
from typing import List

from sloop.core import GraphBuilder, IntentGenerator
from sloop.models import TaskSkeleton, UserIntent
from sloop.utils import logger, setup_logging


def load_skeletons(path: str) -> List[TaskSkeleton]:
    """从 JSON 文件加载并反序列化为 TaskSkeleton 对象"""
    file_path = Path(path)
    if not file_path.exists():
        logger.error(f"Skeleton file not found: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 使用 Pydantic 的 model_validate 批量转换
    # 注意：确保 TaskSkeleton 的字段定义与 JSON 结构匹配
    try:
        skeletons = [TaskSkeleton.model_validate(item) for item in data]
        logger.info(f"Loaded {len(skeletons)} skeletons from {file_path}")
        return skeletons
    except Exception as e:
        logger.error(f"Failed to parse skeletons: {e}")
        return []


def save_intents(intents: List[UserIntent], filename: str):
    """保存生成的 UserIntent 到文件"""
    output_dir = Path("data/intents")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    # Pydantic 转 Dict
    data_to_save = [
        intent.model_dump(by_alias=True, exclude_none=True) for intent in intents
    ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(intents)} user intents to {path}")


def main():
    setup_logging()

    # 1. 路径配置
    GRAPH_CHECKPOINT = "data/graph_checkpoint.pkl"  # 或绝对路径
    SKELETON_INPUT = "data/samples/skeletons_neighborhood.json"

    # 2. 初始化 GraphBuilder (作为 Knowledge Base)
    # Generator 需要查阅工具的 Description 和 Parameters 才能编造故事
    logger.info("Loading Tool Registry from GraphBuilder...")
    builder = GraphBuilder()
    if not builder.load_checkpoint(GRAPH_CHECKPOINT):
        logger.error(
            "Please run graph_builder_exp.py first to generate the checkpoint."
        )
        return

    # 3. 初始化 IntentGenerator
    logger.info("Initializing Intent Generator (Storyteller Mode)...")
    # 传入 tool_registry (builder.tools)
    generator = IntentGenerator(tool_registry=builder.tools)

    # 4. 加载骨架
    skeletons = load_skeletons(SKELETON_INPUT)
    if not skeletons:
        return

    # 5. 开始批量生成
    logger.info(f"\n>>> Start Generating Intents for {len(skeletons)} Skeletons <<<\n")

    generated_intents = []

    for i, skel in enumerate(skeletons):
        logger.info(f"Processing Skeleton #{i + 1} (Pattern: {skel.pattern})...")

        # 显示核心链条，方便观察
        core_chain = " -> ".join([n.name for n in skel.get_core_nodes()])
        logger.debug(f"Target Chain: {core_chain}")

        # === 核心调用 ===
        intent = generator.generate(skel)

        if intent:
            generated_intents.append(intent)

            # === 打印“Deep Dive”分析，验证生成质量 ===
            logger.info(f"[Intent #{i + 1} Generated]")
            logger.info(f'  Query: "{intent.query}"')
            logger.info(f"  Input (Initial State): {intent.initial_state}")
            logger.info(f"  Goal  (Final State):   {intent.final_state}")
            logger.info("-" * 50)
        else:
            logger.warning(f"Failed to generate intent for Skeleton #{i + 1}")

    # 6. 保存结果
    if generated_intents:
        save_intents(generated_intents, "intents_neighborhood.json")
        logger.success(
            f"Successfully generated {len(generated_intents)}/{len(skeletons)} intents."
        )
    else:
        logger.warning("No intents were generated.")


if __name__ == "__main__":
    main()
