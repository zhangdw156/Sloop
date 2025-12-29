# Sloop: 高保真有状态多轮工具调用数据生成框架 - 技术设计文档（优化版）

**版本:** 0.2.0 (Enhanced)

**日期:** 2025-12-29

**开发语言:** Python (使用 uv 管理)

**核心范式:** 蓝图驱动模拟 (Blueprint-Driven Simulation) + 有限状态机 (FSM)

---

## 1. 项目概述 (Project Overview)

Sloop 是一个下一代合成数据生成框架，旨在解决当前 Agent 训练数据中“逻辑断层”和“无状态幻觉”的痛点。不同于传统的单次 Prompt 生成，Sloop 采用 **"先蓝图，后演绎" (Blueprint-then-Simulate)** 的双阶段架构。

### 1.1 核心设计哲学 (Core Design Philosophy)

- **有状态环境 (Stateful Environment):** 模拟真实的 API 状态变更（如：库存扣减、订单状态流转），而非静态的文本补全。

- **蓝图驱动 (Blueprint-Driven):** 在对话发生前，先生成包含“用户意图”和“标准执行路径（Ground Truth）”的任务蓝图，确保多轮对话有明确的终点。

- **参数级图谱 (Parameter-Level Graph):** 基于工具参数的语义相似度构建依赖关系，而非仅依靠工具描述，确保生成的工具调用链（Tool Chain）在逻辑上可执行。

- **严格质检 (Rigorous Filtering):** 引入 Critic Agent 进行回合级（Turn-level）和轨迹级（Trajectory-level）的双重校验。

---

## 2. 系统架构 (System Architecture)

系统分为三个核心阶段：**构建 (Build)** -> **蓝图 (Blueprint)** -> **模拟 (Simulate)**。

### 2.1 架构分层图 (Architecture Layer Diagram)

```mermaid

graph TD
        subgraph Phase 1: 知识构建
        TB[Tool Graph Builder] -->|解析 & 向量化| KG[参数依赖图谱]
        end

        subgraph Phase 2: 蓝图生成 (The Blueprint)
        KG -->|随机游走/子图采样| TC[Tool Chain Candidates]
        TC -->|Planner Agent| BP[任务蓝图 (Intent + Ground Truth Actions + Expected State)]
        end

        subgraph Phase 3: 交互模拟 (The Loop)
        BP -->|输入| FSM[FSM Engine]
        FSM <-->|交互| UA[User Agent]
        FSM <-->|交互| AA[Assistant Agent]
        FSM <-->|执行 & 更新| SA[Stateful Service Agent]
        end

        subgraph Phase 4: 质量收敛
        FSM -->|原始轨迹| CA[Critic Agent]
        CA -->|校验通过| DB[(最终数据集 JSONL)]
        CA -->|校验失败| DL[丢弃/负样本库]
        end
    
```

---

## 3. 核心组件详设 (Core Components)

### 3.1 预处理：参数级图谱构建器 (Preprocessing: Parameter-Level Tool Graph Builder)

**功能:** 解析 tools.json。

**增强逻辑:** 不仅建立工具间的连接，更建立 Output Parameter (Tool A) -> Input Parameter (Tool B) 的连接。

**算法:** 使用 Embedding 模型计算参数描述的语义相似度，构建有向加权图。

**示例:** search_hotel 的输出 hotel_id 与 book_room 的输入 hotel_id 相似度 > 0.9，建立强连接边。

### 3.2 蓝图生成器 (Blueprint Generator)

**职责:** 在正式对话前，生成“剧本大纲”。

**输出对象 (Blueprint):**

```python

class Blueprint:
        intent: str          # 用户的高层意图（e.g. “我想买一张去北京的票”）
        required_tools: List # 必须使用的工具列表
        ground_truth: List   # 预期的正确调用序列 [search_ticket, book_ticket]
        initial_state: Dict  # 环境初始状态（e.g. {"ticket_count": 1}）
        expected_state: Dict # 结束时期望状态（e.g. {"ticket_count": 0}）
    
```

### 3.3 有状态服务模拟器 (Stateful Service Simulator)

这是区别于普通生成器的关键。它维护一个内存中的 Virtual Environment。

**输入:** tool_call (name, args), current_env_state

**行为:**

- Validate: 校验参数格式。

- Execute: 根据逻辑读取或修改 env_state。

- Response: 返回符合当前状态的结果。

**场景示例:**

- Turn 1: User 查询库存。Service 读取 state['stock']=1，返回 “1”。

- Turn 2: User 购买。Service 修改 state['stock']=0，返回 “Success”。

- Turn 3: User 再次查询。Service 读取 state['stock']=0，返回 “0”。

**优势:** 避免了 LLM 瞎编导致的“无限库存”逻辑漏洞。

### 3.4 裁判智能体 (Critic Agent / Filter)

**职责:** 在生成结束后进行事后审计。

**校验维度:**

|校验维度|说明|
|---|---|
|目标达成率|对话结束时的 final_env_state 是否匹配蓝图中的 expected_state？|
|逻辑连贯性|Assistant 的回复是否基于 Tool Response 的事实？|
|幻觉检测|是否调用了不存在的工具或臆造了参数？|
---

## 4. 详细工作流 (Workflow Specification)

### 4.1 阶段 1: 蓝图构建 (Blueprint Phase)

- Graph Walk: 在工具图谱中随机游走，生成一条合法的工具调用链（e.g. A -> B -> C）。

- Scenario Generation: Planner Agent 根据调用链，反向生成用户的 Intent 和 User Persona（用户画像）。

- State Setup: 初始化虚拟环境数据（如生成虚拟的订单数据库、天气数据）。

### 4.2 阶段 2: 模拟循环 (Simulation Loop - FSM)

- **STATE: INIT**
加载蓝图，重置 Turn Count = 0。

- **STATE: USER_ACTION**
User Agent 基于 Intent 和 History 发言。
Trick: 引入 Best-of-N 采样，让 User Agent 更像真人，不仅仅是复读 Intent。

- **STATE: ASSISTANT_THINK**
Assistant Agent 接收消息，决定 Action。

- **STATE: TOOL_EXECUTION (Stateful)**
Service Agent 接收调用。
关键动作: new_state, response = service.execute(action, current_state)
更新全局环境状态。
记录 Observation。

- **STATE: EVALUATION**
检查是否达到最大轮数。
检查 User 是否发出停止信号（e.g. “###STOP###”）。

### 4.3 阶段 3: 过滤与序列化 (Filtering & Serialization)

- Turn-level Filter: 检查每一轮的 Tool Call 参数是否符合 Schema。

- Trajectory-level Filter: Critic Agent 对比 Actual Execution Path 和 Blueprint Ground Truth。

- Save: 仅保存校验通过的轨迹。

---

## 5. 项目结构规划 (Directory Structure)

```plaintext

Sloop/
    ├── pyproject.toml
    ├── src/
    │   └── sloop/
    │       ├── __init__.py
    │       ├── main.py
    │       ├── config.py
    │       ├── engine/
    │       │   ├── fsm.py           # 有限状态机主循环
    │       │   ├── blueprint.py     # 蓝图生成逻辑 (Phase 2)
    │       │   └── graph.py         # 参数级图谱构建 (Phase 1)
    │       ├── agents/
    │       │   ├── planner.py       # 负责生成 Intent 和 Persona
    │       │   ├── user.py          # 模拟用户
    │       │   ├── assistant.py     # 被测模型
    │       │   ├── service.py       # Stateful Service (含 Mock 逻辑)
    │       │   └── critic.py        # 质量过滤器
    │       └── models/
    │           ├── schema.py        # Pydantic Models
    │           └── state.py         # 定义环境状态结构
    └── data/
        ├── input/                   # tools.json
        └── output/                  # result.jsonl
    
```

---

## 6. 下一步行动建议 (Next Steps)

- 优先实现 graph.py: 编写一个脚本，计算 tool 参数的 Embedding 相似度，这是生成复杂任务链的基础。

- 定义通用的 EnvState 类: 让 Service Agent 能够读写一个共享的字典，而非仅仅返回静态字符串。

- 集成 CrewAI: 将上述 Python 逻辑封装为 CrewAI 的 Custom Tools 或利用 CrewAI 的 Flow 功能来编排。

## 7. 建议技术栈

### A. 核心基础设施 (Infrastructure)

语言版本: Python 3.12 (为了更好的 Type Hint 支持)

包管理: uv (已定，极速且现代)

环境变量管理: python-dotenv

### B. 智能体与大模型 (Agents & LLM)

Agent 框架: crewai

理由: 你已经选定。适合角色扮演和任务分发。

模型抽象层: litellm

理由: 必选。Sloop 需要支持 OpenAI, Claude, DeepSeek 等多种模型。CrewAI 底层支持它，但建议显式使用它来管理 API Key 和 Model Fallback。

Prompt 管理: jinja2

理由: 不要把 Prompt 硬编码在 Python 字符串里。用模板引擎管理复杂的 System Prompt（特别是包含 Few-Shot 示例时）。

### C. 核心引擎 (Core Engine)

有限状态机: transitions

理由: Python 最流行的 FSM 库。轻量级，支持回调函数（比如进入 TOOL_EXECUTION 状态时自动触发 Service Agent）。

数据验证: pydantic (v2)

理由: Sloop 的基石。用于定义 Blueprint, ToolCall, EnvState 等所有数据结构，确保结构化输出的稳定性。

图谱与算法: networkx

理由: 用于构建工具依赖图，计算最短路径或随机游走。

### D. 数据与检索 (Data & RAG)

向量数据库: chromadb (或简单的 numpy + scikit-learn)

理由: 用于计算工具参数的 Embedding 相似度。对于 <10k 个工具，直接用内存计算（numpy）最快，不需要起 heavy 的数据库服务。

Embedding 模型: sentence-transformers (本地) 或 openai API。

### E. 测试与工具 (Dev Tools)

测试框架: pytest

CLI 界面: typer 或 click (用于构建 sloop run --config... 命令行)

日志: loguru (比 logging 更好用，支持彩色日志，方便调试对话流)

## 8. 开发路线

建议采用 MVP (Minimum Viable Product) 迭代模式，周期约为 5 个 Sprint (周)。

Sprint 1: 骨架搭建与数据定义 (Foundation & Schema)
目标: 跑通“读取 tools.json -> 解析 -> 验证”的流程。

环境初始化: 使用 uv init sloop，配置 pyproject.toml。

Schema 定义: 在 src/sloop/models/ 下定义核心 Pydantic 模型：

ToolDefinition (OpenAI 格式)

ConversationState (包含 history, turn_count)

Blueprint (Intent, GroundTruth)

图谱构建器原型:

实现 src/sloop/engine/graph.py。

任务: 编写脚本，读取 tools.json，提取所有参数名，计算 Embedding，使用 networkx 建立简单的连接。

验收标准: 能输入 book_ticket，输出它依赖的前置工具 query_ticket。

Sprint 2: 蓝图生成器 (The Blueprint)
目标: 生成合理的任务剧本，不涉及对话。

Agent 接入: 配置 CrewAI 的 Planner Agent。

Prompt 编写: 编写“根据工具链生成用户意图”的 Prompt。

蓝图管道:

输入: 工具列表。

过程: Graph Random Walk -> 选出工具链 -> Planner Agent 生成意图。

输出: Blueprint 对象 (JSON)。

验收标准: 生成 10 个蓝图，人工检查意图和工具链是否逻辑自洽。

Sprint 3: 状态机与核心循环 (FSM Engine)
目标: 实现 User 和 Assistant 的“干聊”，暂不包含复杂的 State Mock。

FSM 实现: 使用 transitions 库在 src/sloop/engine/fsm.py 实现状态流转逻辑。

角色扮演: 实现 User Agent (基于 CrewAI) 和 Assistant Agent。

联调:

User 发起提问 -> Assistant 返回 Tool Call -> (暂时 Mock 一个静态返回值) -> Assistant 总结。

验收标准: 能够跑通一个完整的对话 Loop，生成符合格式的 dataset.jsonl（虽然工具返回值可能是假的）。

Sprint 4: 有状态服务模拟 (Stateful Service - The Hard Part)
目标: 让工具返回值具备逻辑一致性。

EnvState 设计: 定义一个通用的 KV 存储 (Dict) 作为虚拟环境。

Service Agent 升级:

接入 LLM，Prompt: "Given current state X and tool call Y, update state and return output".

或者实现简单的 Python 代码执行器 (可选)。

集成测试:

测试“买票”场景：库存 1 -> 买票 -> 库存 0 -> 再买 -> 报错。

验收标准: 生成的数据中，Assistant 不会产生事实性幻觉。

Sprint 5: 质检与交付 (Filtering & Polish)
目标: 提升数据质量，剔除废品。

Critic Agent: 实现后处理逻辑，检查 final_state 是否符合 expected_state。

CLI 封装: 使用 typer 封装入口，支持 sloop generate --input tools.json --count 100。

并发优化: 使用 Python asyncio 或 concurrent.futures 并行生成数据（因为 LLM IO 耗时很长）。