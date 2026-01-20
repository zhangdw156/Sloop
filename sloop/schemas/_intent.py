import hashlib
import json
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, model_validator

# =========================================================================
# 1. 定义基础类型 (必须写在类定义之前)
# =========================================================================
# 限制 State 的 Value 只能是简单类型 (扁平化约束)
PrimitiveType = Union[str, int, float, bool]


# =========================================================================
# 2. 核心实体定义
# =========================================================================
class UserIntent(BaseModel):
    """
    UserIntent: 定义一个任务的“题目” (Task Problem Definition)。
    核心思想：用户的需求 = 变量状态从 Initial 变更为 Final 的过程。
    """

    # 1. ID: 设为 Optional，允许实例化时自动生成
    id: Optional[str] = Field(
        None, description="唯一标识符。如果不传，将根据内容自动生成 MD5"
    )

    # === 2. 自然语言表达 (The Prompt) ===
    query: str = Field(..., description="用户发出的自然语言指令")

    # === 3. 初始状态 (Start State / Initial HashMap) ===
    # 任务开始前，环境中已知的信息 (用户显式提供的实体)
    # 使用 PrimitiveType 限制只能存简单值
    initial_state: Dict[str, PrimitiveType] = Field(
        default_factory=dict,
        description="用户提供的初始变量池 (Flat Key-Value)，例如 {'ip': '1.2.3.4'}",
    )

    # === 4. 目标状态 (Goal State / Final HashMap) ===
    # 任务完成后，Agent 应当获取或产生的变量 (任务完成标准)
    final_state: Dict[str, PrimitiveType] = Field(
        default_factory=dict,
        description="期望达成的最终变量池 (Flat Key-Value)，例如 {'city': 'Tokyo'}",
    )

    # === 5. 上下文约束 (Context) ===
    # 只有在这个工具箱下，上述的 State 转换才是可解的
    available_tools: List[str] = Field(
        ..., description="可用的工具名称列表 (Core + Distractors)"
    )

    # === 6. 元数据 (Meta) ===
    meta: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ensure_id_exists(self) -> "UserIntent":
        """
        Pydantic 验证器：如果初始化时没有提供 ID，则基于内容自动生成确定性 Hash。
        这样既允许外部传入 Skeleton ID，也允许自动去重。
        """
        if not self.id:
            # 序列化关键内容：Query + 排序后的 Initial/Final Keys
            # 注意：我们通常只 Hash Keys，因为 Values 可能是 LLM 编造的，微小的变化不应视为不同的意图
            content = {
                "q": self.query,
                "tools": sorted(self.available_tools),  # 工具箱不同，意图也不同
                "init_keys": sorted(self.initial_state.keys()),
                "final_keys": sorted(self.final_state.keys()),
            }
            # 生成 MD5
            # ensure_ascii=False 保证中文 query 的 hash 一致性
            raw_str = json.dumps(content, sort_keys=True, ensure_ascii=False)
            self.id = hashlib.md5(raw_str.encode()).hexdigest()
        return self
