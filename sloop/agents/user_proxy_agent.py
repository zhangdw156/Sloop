import json
from typing import Union, Sequence
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel

from sloop.models import UserIntent
from sloop.utils.logger import logger
from sloop.prompts.simulation import USER_PROXY_SYSTEM_PROMPT
from sloop.configs.env import env_config

class UserProxyAgent(AgentBase):
    def __init__(
        self, 
        name: str, 
        intent: UserIntent, 
        max_turns: int = 10,
        **kwargs
    ):
        super().__init__()
        
        self.name = name
        self.memory = InMemoryMemory()
        self.intent = intent
        self.max_turns = max_turns
        self.current_turn = 0

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
                "temperature": 1.0,
                "max_tokens": 1024
            },
            stream=False 
        )

        sys_content = USER_PROXY_SYSTEM_PROMPT.format(
            query=intent.query,
            initial_state=json.dumps(intent.initial_state, ensure_ascii=False),
            final_state=json.dumps(intent.final_state, ensure_ascii=False)
        )
        self.sys_prompt_dict = {"role": "system", "content": sys_content}

    @override
    async def reply(self, x: Union[Msg, Sequence[Msg]] = None) -> Msg:
        self.current_turn += 1

        if x:
            if isinstance(x, list): await self.memory.add(x)
            else: await self.memory.add(x)

        if self.current_turn > self.max_turns:
            return Msg(name=self.name, role="user", content="TERMINATE_FAILED")

        if self.current_turn == 1:
            msg = Msg(name=self.name, role="user", content=self.intent.query)
            await self.memory.add(msg)
            return msg

        history_msgs = await self.memory.get_memory()
        
        openai_messages = [self.sys_prompt_dict]
        for m in history_msgs:
            openai_messages.append({
                "role": m.role, 
                "content": str(m.content or "")
            })
        
        response = await self.model(messages=openai_messages)
        
        text_content = ""
        # [修复] 字典访问
        for block in response.content:
            block_type = block.get("type") if isinstance(block, dict) else block.type
            if block_type == "text":
                text_val = block.get("text", "") if isinstance(block, dict) else block.text
                text_content += text_val
        
        if "TERMINATE" in text_content and len(text_content) < 50:
            text_content = "TERMINATE"
        
        msg = Msg(name=self.name, role="user", content=text_content)
        await self.memory.add(msg)
        return msg