import json
from typing import Dict, List, Union, cast, override

from agentscope.agent import AgentBase
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import ChatResponse, OpenAIChatModel

from ..configs import env_config
from ..prompts.simulation import SIMULATOR_SYSTEM_PROMPT, SIMULATOR_USER_PROMPT
from ..schemas import TaskSkeleton, UserIntent
from ..utils import logger


class SimulatorAgent(AgentBase):
    def __init__(self, name: str, intent: UserIntent, skeleton: TaskSkeleton, **kwargs):
        super().__init__()

        self.name = name
        self.memory = InMemoryMemory()
        self.intent = intent
        self.skeleton = skeleton
        self.formatter = OpenAIChatFormatter()
        model_name = env_config.get("OPENAI_MODEL_NAME")
        base_url = env_config.get("OPENAI_MODEL_BASE_URL")
        api_key = env_config.get("OPENAI_MODEL_API_KEY")

        if not model_name or not base_url:
            raise ValueError("Missing model config in .env file!")

        self.model = OpenAIChatModel(
            model_name=model_name,
            api_key=api_key,
            client_kwargs={"base_url": base_url},
            generate_kwargs={
                "temperature": 0.1,
                "max_tokens": 2048,
                "response_format": {"type": "json_object"},
            },
            stream=False,
        )

        core_nodes: List[Dict] = [sk.model_dump() for sk in skeleton.get_core_nodes()]
        sys_prompt_content = SIMULATOR_SYSTEM_PROMPT.format(
            initial_state=json.dumps(intent.initial_state, ensure_ascii=False),
            final_state=json.dumps(intent.final_state, ensure_ascii=False),
            core_nodes=json.dumps(core_nodes, ensure_ascii=False),
        )

        self.sys_msg = Msg(name="system", role="system", content=sys_prompt_content)

    @override
    async def reply(self, x: Union[Msg, List[Msg]] | None = None) -> Msg:
        if x is None:
            return Msg(name=self.name, role="assistant", content="Error: No input.")

        # 处理 Msg 输入类型安全
        msg_in: Msg
        if isinstance(x, list):
            msg_in = x[-1]
        elif isinstance(x, Msg):
            msg_in = x
        else:
            try:
                msg_in = x[-1]  # type: ignore
            except Exception:
                return Msg(
                    name=self.name,
                    role="assistant",
                    content="Error: Invalid input type.",
                )

        # 解析 Tool Calls
        tool_use_blocks = msg_in.get_content_blocks("tool_use")

        if not tool_use_blocks:
            return Msg(
                name=self.name,
                role="assistant",
                content="No tool calls found in message.",
            )

        results = []

        # 遍历生成 Mock 数据
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

        prompt_content = SIMULATOR_USER_PROMPT.format(
            tool_name=tool_name, args_str=args_str
        )

        # 1. 构造 Msg 列表
        input_msgs = [
            self.sys_msg,
            Msg(name="user", role="user", content=prompt_content),
        ]

        # 2. 格式化为 OpenAI 格式
        openai_messages = await self.formatter.format(input_msgs)

        try:
            # 3. 调用模型
            raw_response = await self.model(messages=openai_messages)
            response = cast(ChatResponse, raw_response)

            content_str = response.content[0].get("text", "")

            # --- JSON 提取 ---
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
