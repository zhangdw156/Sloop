"""
Sloop CLI 主入口
"""

import json

import typer

from sloop.core.agents.factory import create_sloop_system
from sloop.core.config import SloopConfig
from sloop.core.generation.service_api import API
from sloop.core.optimization import DataOptimizer
from sloop.core.probing import CapabilityProber

app = typer.Typer(help="Sloop: 一个基于强弱模型闭环的数据优化工具。")


@app.command()
def gen(
    services_file: str = typer.Option(
        "services.json", "--services", "-s", help="服务定义文件路径"
    ),
    output_file: str = typer.Option(
        "dataset.json", "--output", "-o", help="输出数据集文件路径"
    ),
    agent_config: str = typer.Option(
        None, "--agent-config", "-c", help="Agent 配置文件的路径 (YAML)"
    ),
):
    """
    使用强模型生成高质量的服务调用对话数据集。
    """
    config = SloopConfig()
    if not config.validate():
        typer.secho(
            "错误: 请检查 .env 文件中的 API 配置。", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)

    # 1. 使用工厂创建 Sloop 系统
    sloop_system = create_sloop_system(config, agent_config)

    # 3. 读取服务定义
    with open(services_file, "r", encoding="utf-8") as f:
        services_data = json.load(f)
    apis = [API(**service) for service in services_data]

    # 4. 生成数据集
    dataset = []
    # 假设我们生成一个对话作为示例
    conversation = sloop_system.generate_conversation("帮我预定一个餐厅", apis)
    dataset.append(conversation)

    # 5. 保存数据集
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    typer.echo(f"已从 {services_file} 生成数据并保存至 {output_file}。")


@app.command()
def probe(
    dataset_file: str = typer.Option(
        "dataset.json", "--dataset", "-d", help="输入数据集文件路径"
    ),
    output_file: str = typer.Option(
        "boundary_cases.json", "--output", "-o", help="输出边界案例文件路径"
    ),
):
    """
    使用弱模型执行 Greedy Capability Probing (GCP)，识别边界案例。
    """
    config = SloopConfig()
    if not config.validate():
        typer.secho(
            "错误: 请检查 .env 文件中的 API 配置。", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)

    prober = CapabilityProber(config)
    prober.run_probe(dataset_file, output_file)


@app.command()
def optimize(
    original_dataset: str = typer.Option(
        "dataset.json", "--original", "-i", help="原始数据集文件路径"
    ),
    boundary_cases: str = typer.Option(
        "boundary_cases.json", "--boundary", "-b", help="边界案例文件路径"
    ),
    services_file: str = typer.Option(
        "services.json", "--services", "-s", help="服务定义文件路径"
    ),
    output_file: str = typer.Option(
        "optimized_dataset.json", "--output", "-o", help="输出优化后数据集文件路径"
    ),
):
    """
    使用强模型执行 JGLV (标签校验) 和 EDDE (错误驱动扩展)，优化数据集。
    """
    config = SloopConfig()
    if not config.validate():
        typer.secho(
            "错误: 请检查 .env 文件中的 API 配置。", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)

    optimizer = DataOptimizer(config)
    optimizer.optimize_dataset(
        original_dataset, boundary_cases, services_file, output_file
    )


if __name__ == "__main__":
    app()
