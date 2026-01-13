import json
from typing import List, override

from agentscope.agent import AgentBase
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel

from ..configs import env_config
from ..prompts.simulation import USER_PROXY_SYSTEM_PROMPT
from ..schemas import UserIntent


class UserProxyAgent(AgentBase):
    def __init__(self, name: str, intent: UserIntent, max_turns: int = 10, **kwargs):
        super().__init__()

        self.name = name
        self.memory = InMemoryMemory()
        self.intent = intent
        self.max_turns = max_turns
        self.current_turn = 0
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
            generate_kwargs={"temperature": 1.0, "max_tokens": 512},
            stream=False,
        )

        sys_content = USER_PROXY_SYSTEM_PROMPT.format(
            query=intent.query,
            initial_state=json.dumps(intent.initial_state, ensure_ascii=False),
            final_state=json.dumps(intent.final_state, ensure_ascii=False),
        )

        self.sys_msg = Msg(name="system", role="system", content=sys_content)

    @override
    async def reply(self, x: Msg | List[Msg] | None = None) -> Msg:
        self.current_turn += 1

        await self.memory.add(x)

        if self.current_turn > self.max_turns:
            return Msg(name=self.name, role="user", content="TERMINATE_FAILED")

        if self.current_turn == 1:
            msg = Msg(name=self.name, role="user", content=self.intent.query)
            await self.memory.add(msg)
            return msg

        input_msgs = [self.sys_msg] + await self.memory.get_memory()

        # 2. 调用 formatter 生成符合 OpenAI API 标准的 List[Dict]
        openai_messages = await self.formatter.format(input_msgs)

        # 3. 调用模型
        raw_response = await self.model(messages=openai_messages)

        # 结果解析逻辑
        text_content = raw_response.content[0].get("text", 0)

        # 简单的终止判定逻辑
        if "TERMINATE" in text_content:
            text_content = "TERMINATE"

        msg = Msg(name=self.name, role="user", content=text_content)
        await self.memory.add(msg)
        return msg
