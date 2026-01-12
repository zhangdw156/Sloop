import json
from typing import Sequence, Union, cast

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from agentscope.agent import AgentBase
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import ChatResponse, OpenAIChatModel

from sloop.configs.env import env_config
from sloop.models import TaskSkeleton, UserIntent
from sloop.prompts.simulation import SIMULATOR_SYSTEM_PROMPT
from sloop.utils.logger import logger


class SimulatorAgent(AgentBase):
    def __init__(self, name: str, intent: UserIntent, skeleton: TaskSkeleton, **kwargs):
        super().__init__()

        self.name = name
        self.memory = InMemoryMemory()
        self.intent = intent
        self.skeleton = skeleton

        model_name = env_config.get("OPENAI_MODEL_NAME")
        base_url = env_config.get("OPENAI_MODEL_BASE_URL")
        api_key = env_config.get("OPENAI_MODEL_API_KEY") or "EMPTY"

        if not model_name or not base_url:
            raise ValueError("Missing model config in .env file!")

        self.model = OpenAIChatModel(
            model_name=model_name,
            api_key=api_key,
            client_kwargs={"base_url": base_url},
            generate_kwargs={
                "temperature": 0.01,
                "max_tokens": 2048,
                "response_format": {"type": "json_object"},
            },
            # 必须为 False，确保返回 ChatResponse 对象而不是生成器
            stream=False,
        )

        core_nodes = [n.name for n in skeleton.get_core_nodes()]
        self.base_sys_prompt = SIMULATOR_SYSTEM_PROMPT.format(
            initial_state=json.dumps(intent.initial_state, ensure_ascii=False),
            final_state=json.dumps(intent.final_state, ensure_ascii=False),
            core_nodes=json.dumps(core_nodes, ensure_ascii=False),
        )

    @override
    async def reply(self, x: Union[Msg, Sequence[Msg]] | None = None) -> Msg:
        if x is None:
            return Msg(name=self.name, role="assistant", content="Error: No input.")

        # [修复 1] 处理 Sequence[Msg] 类型报错
        # 显式判断并提取单个 Msg 对象，消除类型歧义
        msg_in: Msg
        if isinstance(x, list):
            msg_in = x[-1]
        elif isinstance(x, Msg):
            msg_in = x
        else:
            # 处理 tuple 或其他 Sequence 情况
            try:
                msg_in = x[-1]  # type: ignore
            except Exception:
                return Msg(
                    name=self.name,
                    role="assistant",
                    content="Error: Invalid input type.",
                )

        # 现在 msg_in 确定是 Msg 类型，可以安全调用 get_content_blocks
        tool_use_blocks = msg_in.get_content_blocks("tool_use")

        if not tool_use_blocks:
            return Msg(
                name=self.name,
                role="assistant",
                content="No tool calls found in message.",
            )

        results = []

        for block in tool_use_blocks:
            tool_name = block.get("name")
            tool_args = block.get("input", {})
            tool_args_str = json.dumps(tool_args, ensure_ascii=False)

            mock_data_str = await self._generate_mock_observation_with_llm(
                tool_name, tool_args_str
            )
            results.append(mock_data_str)

        final_content = "\n".join(results)

        return Msg(name=self.name, role="assistant", content=final_content)

    async def _generate_mock_observation_with_llm(
        self, tool_name: str, args_str: str
    ) -> str:
        logger.info(f"LLM Simulator generating for: {tool_name} args={args_str}")

        prompt_content = (
            f"Please generate a JSON response for the tool: '{tool_name}'.\n"
            f"Input arguments: {args_str}\n"
            f"Ensure the response is consistent with the User Intent and Final State provided in the system prompt.\n"
            f"Return ONLY valid JSON."
        )

        messages = [
            {"role": "system", "content": self.base_sys_prompt},
            {"role": "user", "content": prompt_content},
        ]

        try:
            raw_response = await self.model(messages=messages)

            response = cast(ChatResponse, raw_response)

            content_str = ""
            if hasattr(response, "content"):
                for block in response.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content_str += str(block.get("text", ""))
            else:
                # 兜底：万一它是字符串
                content_str = str(response)

            # --- 下面是通用的 JSON 提取逻辑 ---
            clean_content = (
                content_str.replace("```json", "").replace("```", "").strip()
            )

            start = clean_content.find("{")
            end = clean_content.rfind("}") + 1
            if start != -1 and end != -1:
                json_str = clean_content[start:end]
                json.loads(json_str)  # 校验
                return json_str
            else:
                logger.warning(f"Simulator LLM output invalid JSON: {content_str}")
                return json.dumps({
                    "status": "error",
                    "message": "Simulator generation format error",
                    "raw": content_str,
                })

        except Exception as e:
            logger.error(f"Simulator LLM failed: {e}")
            return json.dumps({
                "status": "error",
                "message": "Simulation internal error",
            })
