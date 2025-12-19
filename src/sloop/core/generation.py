"""
Sloop 核心功能模块。
"""

import typer
from typing import Optional


app = typer.Typer(help="Sloop: 一个用于生成和优化工具调用数据集的 CLI 工具。")


@app.command()
def gen(
    input_file: str = typer.Option(
        "services.json", "--input", "-i", help="输入的 JSON 文件，包含服务/工具定义。"
    ),
    output_dir: str = typer.Option(
        "output", "--output", "-o", help="生成数据的输出目录。"
    ),
    count: int = typer.Option(1, "--count", "-c", help="生成对话组的数量。"),
    description: Optional[str] = typer.Option(
        None, "--description", "-d", help="用于辅助生成数据的文本描述。"
    ),
):
    """
    利用强模型 API 生成高质量的服务调用对话数据集。
    """
    # TODO: 实现重构后的数据生成逻辑
    # 1. 读取输入文件
    # 2. 组织成 ListAPIStructure
    # 3. 使用 RandomAPISampler 采样
    # 4. 使用 SimpleUserAgent 生成用户画像和问题
    # 5. 使用 SimplePlanner, SimpleAssistantAgent, SimpleServiceAgent 执行多轮对话
    # 6. 保存每个对话为单独的 JSON 文件
    # 7. 汇总所有对话
    typer.echo(f"已从 {input_file} 生成 {count} 组数据并保存至 {output_dir}。")


@app.command()
def probe(
    dataset_file: str = typer.Option(
        "dataset.json", "--dataset", "-d", help="输入的数据集文件。"
    ),
    output_file: str = typer.Option(
        "boundary_cases.json", "--output", "-o", help="输出的边界案例文件。"
    ),
):
    """
    利用弱模型 API 执行 Greedy Capability Probing (GCP)。
    """
    # TODO: 实现探测逻辑
    typer.echo(f"已使用数据集 {dataset_file} 进行探测，边界案例已保存至 {output_file}。")


@app.command()
def optimize(
    dataset_file: str = typer.Option(
        "dataset.json", "--dataset", "-d", help="原始数据集文件。"
    ),
    boundary_cases_file: str = typer.Option(
        "boundary_cases.json", "--boundary", "-b", help="边界案例文件。"
    ),
    output_file: str = typer.Option(
        "optimized_dataset.json", "--output", "-o", help="优化后的数据集文件。"
    ),
):
    """
    利用强模型执行 JGLV (标签校验) 和 EDDE (错误驱动扩展)。
    """
    # TODO: 实现优化逻辑
    typer.echo(
        f"已使用 {dataset_file} 和 {boundary_cases_file} "
        f"优化数据集，结果已保存至 {output_file}。"
    )


if __name__ == "__main__":
    app()
