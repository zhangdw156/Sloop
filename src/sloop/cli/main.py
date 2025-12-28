"""
Sloop CLI 主入口
基于CrewAI的数据生成工具
"""

import json
import typer
from pathlib import Path
from typing import Optional

from sloop.core.config import config
from sloop.core.api_structure import load_apis_from_file
from sloop.core.data_generator import BatchDataGenerator

app = typer.Typer(
    help="Sloop: 基于CrewAI的智能工具调用数据生成器",
    add_completion=False
)


@app.command()
def gen(
    services_file: str = typer.Option(
        "services.json", "--services", "-s",
        help="API服务定义文件路径"
    ),
    output_file: str = typer.Option(
        "dataset.json", "--output", "-o",
        help="输出数据集文件路径"
    ),
    num_conversations: int = typer.Option(
        10, "--num-conversations", "-n",
        help="生成对话数量", min=1, max=1000
    ),
    apis_per_conversation: int = typer.Option(
        3, "--apis-per-conversation", "-k",
        help="每个对话使用的API数量", min=1, max=10
    ),
    target_turns: int = typer.Option(
        10, "--target-turns", "-t",
        help="目标对话轮数（允许±40%偏差）", min=3, max=50
    ),
    sampling_strategy: str = typer.Option(
        "balanced", "--sampling-strategy",
        help="API采样策略 (random/balanced/connected)"
    ),
    structure_type: str = typer.Option(
        "tree", "--structure-type",
        help="API结构化类型 (tree/graph/auto)"
    ),
    verbose: bool = typer.Option(
        True, "--verbose", "-v",
        help="启用详细输出"
    ),
):
    """
    使用CrewAI生成高质量的多轮工具调用对话数据集

    工作流程:
    1. 加载并结构化API定义（树形/图形）
    2. 智能采样相关API组合
    3. 多Agent协作生成对话数据
    4. 输出标准格式的数据集
    """
    # 验证配置
    if not config.validate():
        typer.secho(
            "❌ 配置错误: 请检查 .env 文件中的 SLOOP_STRONG_API_KEY 和 SLOOP_STRONG_BASE_URL",
            fg=typer.colors.RED,
            err=True
        )
        raise typer.Exit(1)

    # 检查输入文件
    services_path = Path(services_file)
    if not services_path.exists():
        typer.secho(
            f"❌ 服务文件不存在: {services_file}",
            fg=typer.colors.RED,
            err=True
        )
        raise typer.Exit(1)

    # 设置verbose
    config.verbose = verbose

    try:
        # 加载API定义
        typer.echo("📚 加载API服务定义...")
        apis = load_apis_from_file(services_file)
        if not apis:
            typer.secho("❌ API文件为空或格式错误", fg=typer.colors.RED)
            raise typer.Exit(1)

        typer.echo(f"✅ 加载了 {len(apis)} 个API定义")

        # 显示API结构信息
        from sloop.core.api_structure import APICollection
        api_collection = APICollection(apis, structure_type)
        structure_info = api_collection.get_structure_info()

        typer.echo(f"🏗️  API结构化类型: {structure_info['type']}")
        if structure_info['type'] == 'tree':
            typer.echo(f"📁 识别出 {len(structure_info['categories'])} 个功能类别: {', '.join(structure_info['categories'][:5])}{'...' if len(structure_info['categories']) > 5 else ''}")
        else:
            typer.echo(f"🔗 图结构: {structure_info['nodes']} 节点, {structure_info['edges']} 边")

        # 初始化数据生成器
        typer.echo("🤖 初始化CrewAI数据生成器...")
        generator = BatchDataGenerator(apis, structure_type)

        # 显示生成计划
        typer.echo(f"🎯 生成计划:")
        typer.echo(f"   • 对话数量: {num_conversations}")
        typer.echo(f"   • 每对话API数: {apis_per_conversation}")
        typer.echo(f"   • 采样策略: {sampling_strategy}")
        typer.echo(f"   • 输出文件: {output_file}")

        # 确认开始生成
        if not typer.confirm("\n🚀 开始生成数据集?", default=True):
            typer.echo("已取消")
            return

        # 生成数据集
        typer.echo("\n⚡ 开始生成对话数据...")
        dataset = generator.generate_dataset(
            num_conversations=num_conversations,
            apis_per_conversation=apis_per_conversation,
            sampling_strategy=sampling_strategy,
            target_turns=target_turns,
            output_file=output_file
        )

        # 显示统计信息
        if dataset:
            total_conversations = len(dataset)
            avg_quality = sum(conv.get('quality_score', 0) for conv in dataset) / total_conversations
            api_usage = {}
            for conv in dataset:
                for api_name in conv.get('apis_used', []):
                    api_usage[api_name] = api_usage.get(api_name, 0) + 1

            typer.echo(f"\n🎉 生成完成!")
            typer.echo(f"📊 统计信息:")
            typer.echo(f"   • 成功生成对话: {total_conversations}")
            typer.echo(f"   • 平均质量评分: {avg_quality:.2f}")
            typer.echo(f"   • API使用频率: {dict(sorted(api_usage.items(), key=lambda x: x[1], reverse=True)[:5])}")
            typer.echo(f"💾 数据已保存至: {output_file}")
        else:
            typer.secho("❌ 生成失败: 未产生任何对话数据", fg=typer.colors.RED)

    except Exception as e:
        typer.secho(f"❌ 生成过程中出现错误: {e}", fg=typer.colors.RED, err=True)
        if verbose:
            import traceback
            typer.echo(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def analyze(
    services_file: str = typer.Option(
        "services.json", "--services", "-s",
        help="API服务定义文件路径"
    ),
    structure_type: str = typer.Option(
        "auto", "--structure-type",
        help="API结构化类型 (tree/graph/auto)"
    ),
):
    """
    分析API服务定义，显示结构化信息
    """
    try:
        apis = load_apis_from_file(services_file)
        api_collection = APICollection(apis, structure_type)
        structure_info = api_collection.get_structure_info()

        typer.echo("📊 API分析结果:"        typer.echo(f"   • 总API数量: {structure_info['total_apis']}")
        typer.echo(f"   • 结构类型: {structure_info['type']}")

        if structure_info['type'] == 'tree':
            typer.echo(f"   • 功能类别: {len(structure_info['categories'])}")
            for category in structure_info['categories'][:10]:  # 显示前10个
                apis_in_category = len([api for api in apis if api.get('category') == category or
                                      api_collection.structure._extract_category(api) == category])
                typer.echo(f"     - {category}: {apis_in_category} 个API")

        # 显示API详情
        typer.echo("
🔧 API详情:"        for i, api in enumerate(apis[:5], 1):  # 显示前5个
            typer.echo(f"   {i}. {api['name']}: {api.get('description', 'No description')[:50]}...")

        if len(apis) > 5:
            typer.echo(f"   ... 还有 {len(apis) - 5} 个API")

    except Exception as e:
        typer.secho(f"❌ 分析失败: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@app.command()
def validate(
    dataset_file: str = typer.Option(
        ..., "--dataset", "-d",
        help="要验证的数据集文件路径"
    ),
):
    """
    验证生成的数据集格式和质量
    """
    try:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        if not isinstance(dataset, list):
            typer.secho("❌ 数据集格式错误: 应为数组", fg=typer.colors.RED)
            return

        typer.echo(f"📊 数据集验证结果:")
        typer.echo(f"   • 对话数量: {len(dataset)}")

        # 检查格式
        valid_conversations = 0
        total_quality = 0

        for i, conv in enumerate(dataset):
            is_valid = True
            errors = []

            # 检查必需字段
            required_fields = ['conversation', 'label']
            for field in required_fields:
                if field not in conv:
                    is_valid = False
                    errors.append(f"缺少字段: {field}")

            # 检查conversation格式
            if 'conversation' in conv:
                conv_data = conv['conversation']
                if not isinstance(conv_data, list):
                    is_valid = False
                    errors.append("conversation应为数组")
                elif conv_data and not all(isinstance(msg, dict) and 'role' in msg and 'content' in msg for msg in conv_data):
                    is_valid = False
                    errors.append("conversation消息格式错误")

            # 检查label格式
            if 'label' in conv and isinstance(conv['label'], dict):
                label = conv['label']
                if 'tool_call' not in label or 'thought_process' not in label:
                    errors.append("label缺少必需字段")
                else:
                    total_quality += conv.get('quality_score', 0.5)
            else:
                is_valid = False
                errors.append("label格式错误")

            if is_valid:
                valid_conversations += 1
            elif i < 5:  # 只显示前5个错误
                typer.echo(f"   ⚠️ 对话 {i+1} 格式问题: {', '.join(errors)}")

        validity_rate = valid_conversations / len(dataset) * 100
        avg_quality = total_quality / len(dataset)

        typer.echo(f"   • 格式有效率: {validity_rate:.1f}% ({valid_conversations}/{len(dataset)})")
        typer.echo(f"   • 平均质量分: {avg_quality:.2f}")

        if validity_rate >= 95:
            typer.echo("✅ 数据集质量良好")
        elif validity_rate >= 80:
            typer.echo("⚠️ 数据集质量一般，建议检查")
        else:
            typer.secho("❌ 数据集质量较差，需要改进", fg=typer.colors.RED)

    except Exception as e:
        typer.secho(f"❌ 验证失败: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
