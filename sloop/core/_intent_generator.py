import hashlib
import json
from typing import Any, Dict, List

from agentscope.model import OpenAIChatModel

from ..configs import env_config
from ..schemas import TaskSkeleton, ToolDefinition, UserIntent
from ..prompts.generator import (
    INTENT_GENERATOR_SYSTEM_PROMPT,
    INTENT_GENERATOR_USER_TEMPLATE,
)
from ..utils import logger


class IntentGenerator:
    """
    意图生成器：负责根据工具调用骨架，反向推导用户的真实意图。

    Process:
    Skeleton -> IntentGenerator -> UserIntent
    """

    def __init__(self, tool_registry: Dict[str, ToolDefinition]):
        """
        Args:
            tool_registry: 所有可用工具的字典 (用于查阅工具详情)
        """
        self.tool_registry = tool_registry

        base_url = env_config.get("OPENAI_MODEL_BASE_URL")
        api_key = env_config.get("OPENAI_MODEL_API_KEY")
        model_name = env_config.get("OPENAI_MODEL_NAME")

        if not base_url or not api_key:
            logger.warning("Missing LLM configuration in .env. Generator will fail.")
            self.model = None
        else:
            self.model = OpenAIChatModel(
                model_name=model_name,
                api_key=api_key,
                stream=False,
                client_kwargs={"base_url": base_url},
            )

    async def generate(self, skeleton: TaskSkeleton) -> UserIntent | None:
        """
        根据 Skeleton 生成一个完整的 User Intent
        """
        # 1. 准备上下文信息
        core_nodes = skeleton.get_core_nodes()
        if not core_nodes:
            logger.error("Skeleton has no core nodes!")
            return None

        # 2. 构建 Prompt
        tools_desc_str = self._format_tools_desc(core_nodes)
        chain_desc_str = self._format_chain_flow(skeleton)

        user_prompt = INTENT_GENERATOR_USER_TEMPLATE.format(
            tools_desc=tools_desc_str, chain_desc=chain_desc_str
        )

        # 3. 调用 LLM 进行生成
        intent_data = None
        if self.model is None:
            logger.error("LLM model not initialized")
            return None

        response = await self.model(
            messages=[
                {"role": "system", "content": INTENT_GENERATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )

        text = ""
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "").strip()
        if text:
            intent_data = json.loads(text)

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
        tools_json = []
        for node in nodes:
            tool_def = self.tool_registry.get(node.name)
            # 过滤无效工具 + 空描述工具
            if not (tool_def and tool_def.name and tool_def.description):
                continue
            # 构建单工具JSON结构，做容错处理
            single_tool = {
                "name": tool_def.name.strip(),
                "description": tool_def.description.strip(),
                "parameters": tool_def.parameters.model_dump()
                if tool_def.parameters
                else {},
            }
            tools_json.append(single_tool)
        # 序列化为标准JSON字符串，紧凑格式省Token，中文不乱码
        return json.dumps({"tools": tools_json}, ensure_ascii=False)

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
