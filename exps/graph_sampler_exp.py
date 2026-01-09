import json
import os
from collections import Counter
from typing import List

from sloop.models import TaskSkeleton
from sloop.utils import GraphBuilder, GraphSampler, logger, setup_logging


def save_skeletons(skeletons: List[TaskSkeleton], filename: str):
    """辅助函数：保存生成的 TaskSkeleton 到文件"""
    os.makedirs("data/samples", exist_ok=True)
    path = f"data/samples/{filename}"
    
    data_to_save = [skel.model_dump(by_alias=True, exclude_none=True) for skel in skeletons]
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(skeletons)} skeletons to {path}")


def analyze_batch(batch: List[TaskSkeleton], name: str):
    """分析批次数据的分布"""
    if not batch:
        logger.warning(f"Batch {name} is empty.")
        return

    # 使用对象属性访问
    lengths = [len(b.nodes) for b in batch]
    dist = dict(Counter(lengths))

    sorted_dist = dict(sorted(dist.items()))
    logger.info(f"[{name}] Node Count Distribution: {sorted_dist}")

    example = batch[0]

    # 使用对象属性访问
    if example.meta and example.meta.core_chain_nodes:
        # Neighborhood 模式
        core_path = example.meta.core_chain_nodes
        noise_count = len(example.meta.distractor_nodes)
        chain_str = " -> ".join(core_path)
        logger.info(f"[{name}] Core Path: {chain_str} (+ {noise_count} distractors)")
    else:
        # Chain 模式
        # 这里的 edges 是 List[SkeletonEdge] 对象
        edges_str = " -> ".join([f"{e.from_tool}..{e.to_tool}" for e in example.edges])
        logger.info(f"[{name}] Logic Flow: {edges_str}")


def main():
    # 1. 初始化日志
    setup_logging()

    # 2. 加载图谱
    checkpoint_path = "data/graph_checkpoint.pkl" # 建议使用相对路径，或者根据你的环境修改
    builder = GraphBuilder()
    if not builder.load_checkpoint(checkpoint_path):
        logger.error("Failed to load graph checkpoint.")
        return

    builder.show()

    # 3. 初始化采样器
    logger.info("Initializing Graph Sampler (Task Architect Mode)...")
    sampler = GraphSampler(builder.graph)

    # --- 实验阶段 1: 纯线性链采样 (验证 Slot Filling) ---
    logger.info("\n>>> Stage 1: Sequential Chains (Short: 2-3) <<<")

    batch_short = sampler.generate_skeletons(
        mode="chain", count=10, min_len=2, max_len=3
    )
    analyze_batch(batch_short, "ShortChain")
    save_skeletons(batch_short, "skeletons_short.json")

    stats = sampler.get_coverage_stats()
    logger.info(f"Stage 1 Stats: Coverage {stats['coverage_ratio']}")

    # --- 实验阶段 2: 长链采样 (验证 Reasoning) ---
    logger.info("\n>>> Stage 2: Sequential Chains (Long: 4-6) <<<")

    batch_long = sampler.generate_skeletons(
        mode="chain", count=10, min_len=4, max_len=6
    )
    analyze_batch(batch_long, "LongChain")
    save_skeletons(batch_long, "skeletons_long.json")

    stats = sampler.get_coverage_stats()
    logger.info(f"Stage 2 Stats: Coverage {stats['coverage_ratio']}")

    # --- 实验阶段 3: 压力测试 (验证 Decay & Deduplication) ---
    logger.info("\n>>> Stage 3: Stress Test (Chain Mode, Mixed Lengths) <<<")
    logger.info("Generating 1000 unique chain skeletons...")

    batch_stress = sampler.generate_skeletons(
        mode="chain", count=1000, min_len=2, max_len=5
    )

    # 使用对象属性
    lengths = [len(b.nodes) for b in batch_stress]
    logger.info(f"Stress Batch Node Dist: {dict(sorted(Counter(lengths).items()))}")

    save_skeletons(batch_stress, "skeletons_stress.json")

    stats = sampler.get_coverage_stats()
    logger.info(f"Stage 3 Stats: Coverage {stats['coverage_ratio']}")

    # --- 实验阶段 4: 邻域子图采样 (验证 Tool Selection) ---
    logger.info("\n>>> Stage 4: Neighborhood Subgraphs (Core + Noise) <<<")
    logger.info("Generating samples with 50% noise ratio...")

    batch_neighborhood = sampler.generate_skeletons(
        mode="neighborhood", count=5, min_len=2, max_len=4, expansion_ratio=0.5
    )

    if batch_neighborhood:
        analyze_batch(batch_neighborhood, "Neighborhood")
        save_skeletons(batch_neighborhood, "skeletons_neighborhood.json")

        # 详细透视第一条数据
        ex = batch_neighborhood[0]
        # 使用对象属性
        core_nodes = ex.meta.core_chain_nodes
        distractors = ex.meta.distractor_nodes
        all_presented = [n.name for n in ex.nodes]

        logger.info(f"\n[Deep Dive] Skeleton Pattern: {ex.pattern}")
        logger.info(
            f"[Deep Dive] Core Solution ({len(core_nodes)}): {' -> '.join(core_nodes)}"
        )
        logger.info(f"[Deep Dive] Distractors ({len(distractors)}): {distractors}")
        logger.info(
            f"[Deep Dive] Agent Toolbox:\n{json.dumps(all_presented, indent=2)}"
        )
    else:
        logger.warning("Failed to generate neighborhood subgraphs.")

    logger.success("Graph Sampling Experiment Completed Successfully!")


if __name__ == "__main__":
    main()