import json
import os
from collections import Counter
from typing import List, Dict

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

def main():
    # 1. 初始化日志
    setup_logging()
    
    # 2. 加载图谱 (从 Checkpoint 极速加载)
    checkpoint_path = "/dfs/data/work/Sloop/data/graph_checkpoint.pkl"
    builder = ToolGraphBuilder()
    if not builder.load_checkpoint(checkpoint_path):
        logger.error("Failed to load graph checkpoint. Please run tool_graph_exp.py first.")
        return

    # 展示一下图的基本情况
    builder.show()
    
    # 3. 初始化采样器
    logger.info("Initializing Graph Sampler...")
    sampler = GraphSampler(builder.graph)
    
    # --- 实验阶段 1: 混合模式采样 (验证 DAG 结构) ---
    logger.info(">>> Stage 1: Mixed Pattern Sampling (Smoke Test) <<<")
    # 采样 20 条，足够覆盖 sequential, branching, parallel 三种模式
    batch_1 = sampler.generate_blueprints(count=20)
    
    if batch_1:
        # 统计分布情况
        patterns = [b.get('pattern', 'unknown') for b in batch_1]
        counts = dict(Counter(patterns))
        logger.info(f"Pattern Distribution: {counts}") 
        # 预期: {'sequential': ~10, 'branching': ~6, 'parallel': ~4}

        # 找到并打印一个 非线性 (DAG) 的例子来展示
        dag_examples = [b for b in batch_1 if b.get('pattern') in ('branching', 'parallel')]
        example = dag_examples[0] if dag_examples else batch_1[0]
        
        logger.info(f"\n--- Example Blueprint ({example.get('pattern', 'unknown').upper()}) ---")
        logger.info(json.dumps(example, indent=2, ensure_ascii=False))
        logger.info("-------------------------------------\n")
        
        save_blueprints(batch_1, "batch_1_mixed_patterns.json")
    
    # 查看当前覆盖率
    stats = sampler.get_coverage_stats()
    logger.info(f"Stage 1 Stats: Covered {stats['visited_edges']}/{stats['total_edges']} edges ({stats['coverage_ratio']})")

    # --- 实验阶段 2: 大批量“衰减”采样 (验证覆盖率提升) ---
    logger.info("\n>>> Stage 2: Stress Test (Coverage Guided Walk) <<<")
    logger.info("Generating 100 more unique blueprints to expand coverage...")
    
    # 增加数量以测试深层覆盖
    batch_2 = sampler.generate_blueprints(count=100)
    save_blueprints(batch_2, "batch_2_decay_test.json")
    
    # 再次查看覆盖率，应该会有显著提升
    stats = sampler.get_coverage_stats()
    logger.info(f"Stage 2 Stats: Covered {stats['visited_edges']}/{stats['total_edges']} edges ({stats['coverage_ratio']})")
    
    # --- 实验阶段 3: 强制重置与对比 (验证记忆功能) ---
    logger.info("\n>>> Stage 3: Reset & Re-sample <<<")
    
    # 重置记忆
    sampler.reset_coverage()
    
    batch_3 = sampler.generate_blueprints(count=10)
    save_blueprints(batch_3, "batch_3_after_reset.json")
    
    stats = sampler.get_coverage_stats()
    logger.info(f"Stage 3 Stats (Reset): {stats['coverage_ratio']}")

    logger.success("Graph Sampling Experiment Completed Successfully!")

if __name__ == "__main__":
    main()