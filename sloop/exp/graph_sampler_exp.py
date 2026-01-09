import json
import os
from collections import Counter
from typing import Dict, List

from sloop.utils import ToolGraphBuilder, setup_logging
from sloop.utils.graph_sampler import GraphSampler
from sloop.utils import logger


def save_skeletons(skeletons: List[Dict], filename: str):
    """辅助函数：保存生成的 TaskSkeleton 到文件"""
    os.makedirs("data/samples", exist_ok=True)
    path = f"data/samples/{filename}"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(skeletons, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(skeletons)} skeletons to {path}")


def analyze_batch(batch: List[Dict], name: str):
    """分析批次数据的分布"""
    if not batch:
        logger.warning(f"Batch {name} is empty.")
        return

    # [Schema Update] tools_involved -> nodes
    # 统计节点数量
    lengths = [len(b["nodes"]) for b in batch]
    dist = dict(Counter(lengths))

    # 按节点数排序打印分布
    sorted_dist = dict(sorted(dist.items()))
    logger.info(f"[{name}] Node Count Distribution: {sorted_dist}")

    # 打印一条示例
    example = batch[0]
    
    # 根据不同模式展示不同的摘要
    if "meta" in example and "core_chain_nodes" in example["meta"]:
        # Neighborhood 模式：展示核心链 + 噪音数量
        core_path = example["meta"]["core_chain_nodes"]
        noise_count = len(example["meta"]["distractor_nodes"])
        chain_str = " -> ".join(core_path)
        logger.info(f"[{name}] Core Path: {chain_str} (+ {noise_count} distractors)")
    else:
        # Chain 模式：展示完整链条
        # 注意：nodes 列表顺序可能被 shuffle (取决于 sampler 实现)，
        # 最准确的顺序应该从 edges 里的 step 推导，或者直接打印 edges
        # 这里简单起见，打印 edges 的连接关系
        edges_str = " -> ".join([f"{e['from']}..{e['to']}" for e in example['edges']])
        logger.info(f"[{name}] Logic Flow: {edges_str}")


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
    logger.info("Initializing Graph Sampler (Task Architect Mode)...")
    sampler = GraphSampler(builder.graph)

    # --- 实验阶段 1: 纯线性链采样 (验证 Slot Filling) ---
    logger.info("\n>>> Stage 1: Sequential Chains (Short: 2-3) <<<")

    # 使用 mode="chain"
    batch_short = sampler.generate_skeletons(mode="chain", count=10, min_len=2, max_len=3)
    analyze_batch(batch_short, "ShortChain")
    save_skeletons(batch_short, "skeletons_short.json")

    stats = sampler.get_coverage_stats()
    logger.info(f"Stage 1 Stats: Coverage {stats['coverage_ratio']}")

    # --- 实验阶段 2: 长链采样 (验证 Reasoning) ---
    logger.info("\n>>> Stage 2: Sequential Chains (Long: 4-6) <<<")

    batch_long = sampler.generate_skeletons(mode="chain", count=10, min_len=4, max_len=6)
    analyze_batch(batch_long, "LongChain")
    save_skeletons(batch_long, "skeletons_long.json")

    stats = sampler.get_coverage_stats()
    logger.info(f"Stage 2 Stats: Coverage {stats['coverage_ratio']}")

    # --- 实验阶段 3: 压力测试 (验证 Decay & Deduplication) ---
    logger.info("\n>>> Stage 3: Stress Test (Chain Mode, Mixed Lengths) <<<")
    logger.info("Generating 1000 unique chain skeletons...")

    # 使用我们在上一轮实验中验证过的最佳参数 min_len=2
    batch_stress = sampler.generate_skeletons(mode="chain", count=1000, min_len=2, max_len=5)

    lengths = [len(b["nodes"]) for b in batch_stress]
    logger.info(f"Stress Batch Node Dist: {dict(sorted(Counter(lengths).items()))}")

    save_skeletons(batch_stress, "skeletons_stress.json")

    stats = sampler.get_coverage_stats()
    logger.info(f"Stage 3 Stats: Coverage {stats['coverage_ratio']}")

    # --- 实验阶段 4: 邻域子图采样 (验证 Tool Selection) ---
    logger.info("\n>>> Stage 4: Neighborhood Subgraphs (Core + Noise) <<<")
    logger.info("Generating samples with 50% noise ratio...")
    
    # 使用 mode="neighborhood"
    batch_neighborhood = sampler.generate_skeletons(
        mode="neighborhood", 
        count=5, 
        min_len=2, 
        max_len=4, 
        expansion_ratio=0.5
    )
    
    if batch_neighborhood:
        analyze_batch(batch_neighborhood, "Neighborhood")
        save_skeletons(batch_neighborhood, "skeletons_neighborhood.json")
        
        # 详细透视第一条数据，验证结构是否正确
        ex = batch_neighborhood[0]
        core_nodes = ex["meta"]["core_chain_nodes"]
        distractors = ex["meta"]["distractor_nodes"]
        all_presented = [n["name"] for n in ex["nodes"]]
        
        logger.info(f"\n[Deep Dive] Skeleton Pattern: {ex.get('pattern')}")
        logger.info(f"[Deep Dive] Core Solution ({len(core_nodes)}): {' -> '.join(core_nodes)}")
        logger.info(f"[Deep Dive] Distractors ({len(distractors)}): {distractors}")
        logger.info(f"[Deep Dive] Agent Toolbox:\n{json.dumps(all_presented, indent=2)}")
    else:
        logger.warning("Failed to generate neighborhood subgraphs.")

    logger.success("Graph Sampling Experiment Completed Successfully!")


if __name__ == "__main__":
    main()