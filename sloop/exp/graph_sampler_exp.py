import json
import os
from collections import Counter
from typing import Dict, List

from sloop.utils import ToolGraphBuilder, setup_logging
from sloop.utils.graph_sampler import GraphSampler
from sloop.utils.logger import logger


def save_blueprints(blueprints: List[Dict], filename: str):
    """辅助函数：保存生成的蓝图到文件"""
    os.makedirs("data/samples", exist_ok=True)
    path = f"data/samples/{filename}"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(blueprints, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(blueprints)} blueprints to {path}")


def analyze_batch(batch: List[Dict], name: str):
    """分析批次数据的长度分布"""
    if not batch:
        logger.warning(f"Batch {name} is empty.")
        return

    # 统计工具链长度 (tools_involved 的数量)
    lengths = [len(b["tools_involved"]) for b in batch]
    dist = dict(Counter(lengths))

    # 按长度排序打印分布
    sorted_dist = dict(sorted(dist.items()))
    logger.info(f"[{name}] Length Distribution (Nodes): {sorted_dist}")

    # 打印一条具体的路径示例 (Tool A -> Tool B -> ...)
    example = batch[0]
    chain_str = " -> ".join([t["name"] for t in example["tools_involved"]])
    logger.info(f"[{name}] Example Chain: {chain_str}")


def main():
    # 1. 初始化日志
    setup_logging()

    # 2. 加载图谱
    checkpoint_path = "/dfs/data/work/Sloop/data/graph_checkpoint.pkl"
    builder = ToolGraphBuilder()
    if not builder.load_checkpoint(checkpoint_path):
        logger.error("Failed to load graph checkpoint.")
        return

    builder.show()

    # 3. 初始化采样器
    logger.info("Initializing Graph Sampler (Sequential Mode)...")
    sampler = GraphSampler(builder.graph)

    # --- 实验阶段 1: 短链采样 (验证 Slot Filling 能力数据) ---
    logger.info("\n>>> Stage 1: Short Chains (Min=2, Max=3) <<<")

    batch_short = sampler.generate_chains(count=10, min_len=2, max_len=3)
    analyze_batch(batch_short, "Short")
    save_blueprints(batch_short, "chains_short.json")

    stats = sampler.get_coverage_stats()
    logger.info(
        f"Stage 1 Stats: Covered {stats['visited_edges']}/{stats['total_edges']} edges ({stats['coverage_ratio']})"
    )

    # --- 实验阶段 2: 长链采样 (验证 Reasoning / Long Context 数据) ---
    logger.info("\n>>> Stage 2: Long Chains (Min=4, Max=6) <<<")

    batch_long = sampler.generate_chains(count=10, min_len=4, max_len=6)
    analyze_batch(batch_long, "Long")
    save_blueprints(batch_long, "chains_long.json")

    stats = sampler.get_coverage_stats()
    logger.info(
        f"Stage 2 Stats: Covered {stats['visited_edges']}/{stats['total_edges']} edges ({stats['coverage_ratio']})"
    )

    # --- 实验阶段 3: 大规模覆盖率测试 (验证 Decay 算法) ---
    logger.info("\n>>> Stage 3: Coverage Stress Test (Mixed Lengths) <<<")
    logger.info("Generating 1000 mixed chains (Len 3-5) to expand coverage...")

    batch_stress = sampler.generate_chains(count=1000, min_len=2, max_len=5)

    lengths = [len(b["tools_involved"]) for b in batch_stress]
    logger.info(f"Stress Batch Length Dist: {dict(sorted(Counter(lengths).items()))}")

    save_blueprints(batch_stress, "chains_stress_test.json")

    stats = sampler.get_coverage_stats()
    logger.info(
        f"Stage 3 Stats: Covered {stats['visited_edges']}/{stats['total_edges']} edges ({stats['coverage_ratio']})"
    )

    logger.success("Sequential Graph Sampling Experiment Completed!")


if __name__ == "__main__":
    main()
