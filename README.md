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

While inspired by LoopTool's vision, Sloop aims to provide a cleaner, more modular implementation adhering to standard software engineering practices.

### ✨ The Sloop Framework

![Framework](<./assets/framework.png>)

**Key Features:**
- **Strong-Weak Model Closed Loop**: A clear separation of concerns between the strong (teacher) and weak (student) models for data generation, probing, and optimization.
- **Multi-Agent Data Generation**: The `gen` command uses a coordinated system of agents (User, Assistant, Planner, etc.) to generate high-quality, diverse dialogues.
- **Configurable Agent System**: Agent implementations are dynamically loaded from a YAML configuration file, making the system highly extensible.
- **Open-Source Ecosystem**: Designed to work within a cost-effective, open-source environment.

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

Sloop provides a CLI interface. **Currently, only the `gen` command is implemented. Other commands are under active development.**

### 1. Configure Your Environment
Create a `.env` file in the project root based on `.env.example`. You must set the following environment variables for the strong model:
```bash
SLOOP_STRONG_API_KEY=your_strong_api_key
SLOOP_STRONG_BASE_URL=https://api.your-strong-model-provider.com
```

### 2. Prepare Your Service Definitions
Create a `services.json` file that describes the tools/APIs you want to generate data for. Example:
```json
[
  {
    "name": "get_weather",
    "description": "Get the current weather for a location.",
    "parameters": {
      "location": "string"
    }
  }
]
```

### 3. Generate Data with `gen`
Use the `gen` command to generate a dataset of tool-calling dialogues.

```bash
uv run sloop gen --services services.json --output dataset.json --agent-config configs/default_agents.yaml
```

You can customize the agents used by providing a different YAML configuration file.

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
