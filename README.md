<h1 align="center"> <img src="./assets/logo.png" width="270" style="vertical-align:middle;"/><br>Sloop: A Self-Evolving Framework for LLM Tool Calls</a></h1>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python 3.9+">
</p>

<p align="center">
  <a href="#%EF%B8%8F-overview">Overview</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-future-work">Future Work</a> •
  <a href="#-acknowledgement">Acknowledgement</a>
</p>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

## 📣 Latest News
- **[December, 2025]**: 🚀 Sloop is released! The core framework and `gen` command are now available.

## 💡 Overview

Sloop is an open-source framework inspired by LoopTool, designed to close the data-training loop for robust Large Language Model (LLM) tool calls. Our goal is to create a fully automated, model-aware system that iteratively refines both data and models to overcome the limitations of static data pipelines.

Sloop follows a strong-weak model (Teacher-Student) closed-loop paradigm:
- **Strong Model (Teacher API)**: Used for `gen` (generating high-quality initial data) and `optimize` (executing JGLV label correction and EDDE error-driven expansion).
- **Weak Model (Student API)**: The target model to be optimized. Used for `probe` (executing Greedy Capability Probing to identify boundary cases).

### ✨ The New Sloop Framework (v0.2.0)

**基于CrewAI的分层多智能体架构**：

```
高层编排Agent (CrewAI)
├── API分析专家      - 分析API结构
├── 场景规划师       - 设计场景和用户画像
├── 对话协调器       - 协调对话生成
└── 质量评估师       - 评估数据质量

低层对话Agent (核心角色)
├── User Agent       - 模拟用户行为
├── Assistant Agent  - 生成回复和工具调用
└── Service Agent    - 模拟API执行结果
```

**核心创新**：
- **分层Agent架构**: 高层编排vs低层执行的清晰分工
- **用户画像系统**: 7种用户类型，支持多样化对话生成
- **智能API结构化**: 树形/图形组织，支持游走采样
- **多轮对话控制**: 指定目标轮数，生成高质量长对话
- **CrewAI集成**: 专业多Agent协作框架

**Key Features:**
- **Hierarchical Agent System**: 分层设计，实现复杂任务编排
- **User Profile Engine**: 7种用户画像，生成真实对话行为
- **Intelligent Sampling**: 树游走/图连通采样，构造合理API组合
- **Multi-turn Control**: 精确控制对话轮数（±40%偏差）
- **Production Ready**: 完整的CLI工具，支持大规模数据生成

## 🔧 Installation

### Environment Setup
```bash
# Create a new environment using uv (recommended)
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

## 🛠️ Usage

Sloop provides a comprehensive CLI interface with multiple commands for data generation and analysis.

### 1. Configure Your Environment
Create a `.env` file in the project root based on `.env.example`:
```bash
# 强模型配置 (必需) - 用于数据生成
SLOOP_STRONG_API_KEY=your_strong_api_key_here
SLOOP_STRONG_BASE_URL=https://api.strongmodel.com/v1
SLOOP_STRONG_MODEL_NAME=gpt-4o

# 弱模型配置 (可选) - 用于能力探测
SLOOP_WEAK_API_KEY=your_weak_api_key_here
SLOOP_WEAK_BASE_URL=https://api.weakmodel.com/v1
SLOOP_WEAK_MODEL_NAME=gpt-3.5-turbo

# 系统配置
SLOOP_VERBOSE=true
```

**配置说明**:
- `SLOOP_STRONG_*`: 强模型配置，用于生成高质量数据（必需）
- `SLOOP_WEAK_*`: 弱模型配置，用于能力探测（可选）
- `SLOOP_VERBOSE`: 是否启用详细输出（默认true）

### 2. Prepare Your Service Definitions
Create a `services.json` file with your API definitions:
```json
[
  {
    "name": "get_weather",
    "description": "获取指定城市的天气信息",
    "parameters": {
      "city": "string",
      "unit": "string"
    },
    "category": "weather"
  },
  {
    "name": "search_restaurants",
    "description": "搜索餐厅",
    "parameters": {
      "city": "string",
      "cuisine_type": "string"
    },
    "category": "travel"
  }
]
```

### 3. Analyze Your APIs
Before generating data, analyze your API structure:
```bash
# 分析API结构和类别
uv run sloop analyze --services services.json
```

### 4. Generate Training Data
Use the `gen` command with CrewAI-powered multi-agent generation:
```bash
# 基本用法：生成10个对话
uv run sloop gen --services services.json --output dataset.json

# 高级用法：自定义参数
uv run sloop gen \
  --services services.json \
  --output dataset.json \
  --num-conversations 50 \
  --apis-per-conversation 3 \
  --sampling-strategy balanced \
  --structure-type tree \
  --verbose
```

**参数说明**:
- `--num-conversations`: 生成**对话样本**的数量，每样本包含多轮完整对话 (默认10)
- `--apis-per-conversation`: 每个对话样本使用的API数量 (默认3)
- `--target-turns`: 目标对话轮数，允许±40%偏差 (默认10，范围3-50)
- `--sampling-strategy`: API采样策略 (random/balanced/connected/tree_walk)
- `--structure-type`: API组织方式 (tree/graph/auto)

**新增功能**:
- 🎭 **用户画像系统**: 7种不同用户类型（细心、粗心、表达不清、好奇、技术、商务、新手）
- 🧠 **智能采样**: 支持树游走和图连通采样，构造合理的API序列
- 🔄 **对话轮数控制**: 可指定目标轮数，生成高质量多轮对话
- 📊 **复杂场景**: 根据用户画像和采样API生成多样化场景

### 5. Validate Generated Data
Check the quality of your generated dataset:
```bash
# 验证数据集格式和质量
uv run sloop validate --dataset dataset.json
```

### Example Workflow
```bash
# 1. 设置环境
cp .env.example .env
# 编辑.env文件设置API密钥

# 2. 分析API结构
uv run sloop analyze --services services.json

# 3. 生成高质量多轮对话数据
uv run sloop gen \
  --services services.json \
  --output dataset.json \
  --num-conversations 100 \
  --target-turns 10 \
  --apis-per-conversation 3 \
  --sampling-strategy tree_walk \
  --structure-type tree

# 4. 验证生成的数据质量
uv run sloop validate --dataset dataset.json
```

### Advanced Usage Examples

#### 生成特定用户类型的对话
```bash
# 生成技术型用户的对话（关注API细节和错误处理）
uv run sloop gen --services services.json --output tech_dataset.json --user-type technical

# 生成新手用户的对话（基础问题，需要详细指导）
uv run sloop gen --services services.json --output novice_dataset.json --user-type novice
```

#### 使用图结构进行复杂采样
```bash
# 使用图结构和连通采样，生成相关性强的API组合
uv run sloop gen \
  --services services.json \
  --output connected_dataset.json \
  --sampling-strategy connected \
  --structure-type graph \
  --relationships api_relationships.json
```

#### 生成超长对话进行深度测试
```bash
# 生成15-25轮的长对话，测试复杂场景
uv run sloop gen \
  --services services.json \
  --output long_conversations.json \
  --target-turns 20 \
  --num-conversations 50
```

### Output Format
Generated conversations follow this structure:
```json
[
  {
    "id": "conv_0001",
    "problem": "用户的问题描述",
    "apis_used": ["api1", "api2"],
    "conversation": [
      {"role": "user", "content": "用户查询"},
      {"role": "assistant", "content": "助手回复和工具调用"},
      {"role": "tool", "content": "工具执行结果"},
      {"role": "assistant", "content": "最终回复"}
    ],
    "label": {
      "tool_call": {"name": "api_name", "arguments": {...}},
      "thought_process": "推理过程"
    },
    "quality_score": 0.85
  }
]
```

## 🚧 Future Work

The following features are planned for future releases:
- **`probe` Command**: Implement Greedy Capability Probing (GCP) to use the weak model and identify boundary cases.
- **`optimize` Command**: Implement Judgement-Guided Label Verification (JGLV) and Error-Driven Data Expansion (EDDE) using the strong model to refine the dataset.
- **Iterative Loop**: Fully close the loop by using the output of `probe` and `optimize` to generate new training data and retrain the weak model.

## 🙏 Acknowledgement
We are inspired by the excellent work of:
- [LoopTool](https://github.com/zhuiguang-ning/LoopTool)

## 📄 License

This project is released under the [MIT License](LICENSE).

## 📞 Contact

For any questions or feedback, please reach out to us.
