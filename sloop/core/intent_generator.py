import hashlib
import json
from typing import Any, Dict, List

from agentscope.model import OpenAIChatModel

from sloop.configs import env_config
from sloop.models import TaskSkeleton, ToolDefinition, UserIntent
from sloop.prompts.generator import (
    INTENT_GENERATOR_SYSTEM_PROMPT,
    INTENT_GENERATOR_USER_TEMPLATE,
)
from sloop.utils.logger import logger


class IntentGenerator:
    """
    意图生成器：负责根据工具调用骨架，反向推导用户的真实意图。

    Process:
    Skeleton (结构) -> IntentGenerator (填肉) -> UserIntent (Query + States)
    """

    def __init__(self, tool_registry: Dict[str, ToolDefinition]):
        """
        Args:
            tool_registry: 所有可用工具的字典 (用于查阅工具详情)
        """
        self.tool_registry = tool_registry

        # 直接初始化 agentscope 的 OpenAIChatModel
        base_url = env_config.get("OPENAI_MODEL_BASE_URL")
        api_key = env_config.get("OPENAI_MODEL_API_KEY")
        model_name = env_config.get(
            "OPENAI_MODEL_NAME", "Qwen3-235B-A22B-Instruct-2507"
        )

        if not base_url or not api_key:
            logger.warning("Missing LLM configuration in .env. Generator will fail.")
            self.model = None
        else:
            self.model = OpenAIChatModel(
                model_name=model_name or "Qwen3-235B-A22B-Instruct-2507",
                api_key=api_key,
                stream=False,
                client_kwargs={"base_url": base_url},
            )

    async def generate(
        self, skeleton: TaskSkeleton, max_retries: int = 3
    ) -> UserIntent | None:
        """
        根据骨架生成一个完整的 User Intent
        """
        # 1. 准备上下文信息
        core_nodes = skeleton.get_core_nodes()
        if not core_nodes:
            logger.error("Skeleton has no core nodes!")
            return None

        # 2. 构建 Prompt
        # 只提取核心链的工具定义给 LLM 看，减少 Token 消耗，聚焦任务逻辑
        tools_desc_str = self._format_tools_desc(core_nodes)
        chain_desc_str = self._format_chain_flow(skeleton)

        user_prompt = INTENT_GENERATOR_USER_TEMPLATE.format(
            tools_desc=tools_desc_str, chain_desc=chain_desc_str
        )

        # 3. 调用 LLM 进行生成
        intent_data = None
        for attempt in range(max_retries):
            try:
                if self.model is None:
                    logger.error("LLM model not initialized")
                    break

                response = await self.model(
                    messages=[
                        {"role": "system", "content": INTENT_GENERATOR_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7,  # 稍微高一点的温度，保证生成的具体实体（如IP、地名）多样化
                )

                if response and response.content:
                    # 从 ChatResponse 中提取文本
                    text = ""
                    for block in response.content:
                        if hasattr(block, "text"):
                            text += block.text

                    if text:
                        # 清洗 markdown 标记 (以防万一)
                        clean_resp = (
                            text.replace("```json", "").replace("```", "").strip()
                        )
                        intent_data = json.loads(clean_resp)
                        break
            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")

        if not intent_data:
            logger.error("Failed to generate intent after retries.")
            return None

        # 4. 组装并返回 Pydantic 对象
        try:
            # 提取所有工具名称 (包含干扰项) 作为 Context
            all_tool_names = [n.name for n in skeleton.nodes]

            # 使用骨架 ID 作为 Intent ID 的前缀，保证血缘
            # 但不传 id 参数，让 UserIntent 内部逻辑基于 Query 再次 Hash 去重
            sig = skeleton.get_edges_signature()
            skel_id = f"skel_{hashlib.md5(sig.encode()).hexdigest()}"
            intent = UserIntent(
                query=intent_data.get("query"),
                initial_state=intent_data.get("initial_state", {}),
                final_state=intent_data.get("final_state", {}),
                available_tools=all_tool_names,
                meta={
                    "skeleton_id": skel_id,
                    "scenario": intent_data.get("scenario_summary", ""),
                    "pattern": skeleton.pattern,
                    "generated_by": "sloop_v0.2",
                },
            )
            return intent

        except Exception as e:
            logger.error(f"Failed to parse LLM response into UserIntent: {e}")
            logger.debug(f"Raw LLM response: {intent_data}")
            return None

    def _format_tools_desc(self, nodes: List[Any]) -> str:
        """格式化工具描述供 Prompt 使用"""
        lines = []
        for node in nodes:
            tool_def = self.tool_registry.get(node.name)
            if tool_def:
                # 简化格式，重点展示 description 和 parameters
                lines.append(f"--- Tool: {tool_def.name} ---")
                lines.append(f"Description: {tool_def.description}")
                lines.append(
                    f"Parameters: {json.dumps(tool_def.parameters.dict(), indent=2)}"
                )
                lines.append("")
        return "\n".join(lines)

    def _format_chain_flow(self, skeleton: TaskSkeleton) -> str:
        """格式化执行流供 Prompt 使用 (Human Readable)"""
        # 使用 skeleton.edges 里的 step 信息排序
        sorted_edges = sorted(skeleton.edges, key=lambda x: x.step)

        lines = []
        for edge in sorted_edges:
            lines.append(f"Step {edge.step}: {edge.from_tool} -> {edge.to_tool}")
            if edge.dependency.parameter:
                lines.append(
                    f"   (Passes output to parameter: '{edge.dependency.parameter}')"
                )
        return "\n".join(lines)
