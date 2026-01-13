from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Dependency(BaseModel):
    parameter: Optional[str] = None
    relation: str = "provides_input_for"


class SkeletonEdge(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    step: int
    # 使用 alias 允许 Python 属性为 from_tool，序列化 JSON 时为 "from"
    from_tool: str = Field(..., alias="from")
    to_tool: str = Field(..., alias="to")
    dependency: Dependency


class SkeletonNode(BaseModel):
    name: str
    description: str
    category: str = "general"
    role: Literal["core", "distractor"] = "core"


class SkeletonMeta(BaseModel):
    core_chain_nodes: List[str]
    distractor_nodes: List[str] = Field(default_factory=list)


class TaskSkeleton(BaseModel):
    """
    任务骨架实体类
    代表一个抽象的任务逻辑，不包含具体的用户意图(Intent)。
    """

    pattern: Literal["sequential", "neighborhood_subgraph", "chain"]
    nodes: List[SkeletonNode]
    edges: List[SkeletonEdge]
    meta: Optional[SkeletonMeta] = None

    def get_core_nodes(self) -> List[SkeletonNode]:
        """辅助方法：获取核心节点"""
        return [n for n in self.nodes if n.role == "core"]

    def get_edges_signature(self) -> str:
        """辅助方法：生成边的指纹用于去重"""
        # 使用 set 排序，确保无视边的物理顺序
        # 注意使用 from_tool 和 to_tool 属性
        sigs = [f"{e.from_tool}->{e.to_tool}" for e in self.edges]
        return "|".join(sorted(sigs))
