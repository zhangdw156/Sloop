# Sloop

**Sloop** 是一个基于下推自动机 (Pushdown Automaton, PDA) 的多轮工具调用 (Tool-Use) 数据生成框架。它通过模拟 User、Assistant 和 Service 三方的交互，自动生成高质量、逻辑严密的对话数据，用于训练和评估大语言模型的工具调用能力。

## ✨ 核心特性

- **PDA 驱动引擎**: 摒弃传统的有限状态机 (FSM)，采用下推自动机 (PDA) 管理对话状态。利用栈结构处理嵌套的工具调用和复杂的对话流转，状态管理更加灵活强大。
- **多智能体模拟 (Multi-Agent Simulation)**:
  - **User Agent**: 基于意图 (Intent) 和蓝图 (Blueprint) 模拟用户行为。
  - **Assistant Agent**: 模拟被测模型，执行思考 (Think)、决策 (Decide) 和工具调用 (Tool Call)。
  - **Service Agent**: 模拟真实 API 服务，维护环境状态 (Environment State)，返回执行结果。
- **智能蓝图生成 (Blueprint Generation)**: 结合工具图谱 (Tool Graph) 分析和 LLM 的想象力，自动生成合理的对话意图和任务蓝图。
- **工具图谱构建**: 使用 NetworkX 构建参数级的工具依赖图，确保生成的工具调用序列符合逻辑依赖。
- **全异步/流式支持**: 核心组件设计支持异步操作（具体实现视 LLM 客户端而定）。

## 🛠️ 安装指南

本项目推荐使用 [uv](https://github.com/astral-sh/uv) 进行依赖管理和环境配置。

### 前置要求

- Python >= 3.12
- uv (可选，但推荐)

### 使用 uv 安装 (推荐)

```bash
# 克隆项目
git clone https://github.com/your-org/Sloop.git
cd Sloop

# 同步依赖环境
uv sync

# 激活虚拟环境
source .venv/bin/activate
```

### 使用 pip 安装

```bash
pip install .
```

## 🚀 快速开始

### 1. 配置环境变量

复制示例配置文件并填入你的 LLM API 密钥：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```ini
OPENAI_API_KEY=your_api_key_here
MODEL_NAME=gpt-4o-mini  # 或其他兼容 OpenAI 接口的模型
# OPENAI_API_BASE=...   # 如果使用自定义端点
```

### 2. 运行生成命令

使用 `sloop` 命令行工具生成数据：

```bash
# 查看帮助
sloop --help

# 运行主程序 (具体子命令请参考 help)
sloop
```

*(注：当前 CLI 入口位于 `sloop/cli/main.py`，具体命令参数请以实际帮助信息为准)*

## 📂 项目结构

```text
sloop/
├── agents/             # 智能体模拟器
│   ├── user.py         # 用户模拟
│   ├── assistant.py    # 助手模拟
│   └── service.py      # 服务/环境模拟
├── engine/             # 核心引擎
│   ├── pda.py          # 下推自动机 (PDA) 实现
│   ├── blueprint.py    # 蓝图生成器
│   └── graph.py        # 工具图谱构建
├── models/             # 数据模型 (Pydantic)
│   ├── schema.py       # 基础数据结构
│   ├── state.py        # 状态管理
│   └── blueprint.py    # 蓝图定义
├── utils/              # 工具函数
│   ├── llm.py          # LLM 调用封装
│   └── template.py     # Prompt 模板
├── templates/          # Jinja2 模板文件
└── cli/                # 命令行接口
```

## 🤝 贡献

欢迎提交 Pull Request 或 Issue！在提交代码前，请确保通过了 lint 检查：

```bash
uv run ruff check .
```

## 📄 许可证

[License Name]
