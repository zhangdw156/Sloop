import json
from typing import Union, Sequence, List
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel

from sloop.models import UserIntent, TaskSkeleton
from sloop.utils.logger import logger
from sloop.prompts.simulation import SIMULATOR_SYSTEM_PROMPT
from sloop.configs.env import env_config

class SimulatorAgent(AgentBase):
    def __init__(
        self, 
        name: str, 
        intent: UserIntent, 
        skeleton: TaskSkeleton, 
        **kwargs
    ):
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
                "response_format": {"type": "json_object"} 
            },
            stream=False 
        )

        core_nodes = [n.name for n in skeleton.get_core_nodes()]
        self.base_sys_prompt = SIMULATOR_SYSTEM_PROMPT.format(
            initial_state=json.dumps(intent.initial_state, ensure_ascii=False),
            final_state=json.dumps(intent.final_state, ensure_ascii=False),
            core_nodes=json.dumps(core_nodes, ensure_ascii=False)
        )

    @override
    async def reply(self, x: Union[Msg, Sequence[Msg]] = None) -> Union[Msg, List[Msg]]:
        if x is None:
            return Msg(name=self.name, role="tool", content="Error: No input.")

        msg_in = x[-1] if isinstance(x, list) else x
        tool_calls = getattr(msg_in, "tool_calls", None)
        
        if not tool_calls:
            return Msg(name=self.name, role="tool", content="No tool calls found.")

        response_msgs = []
        
        for call in tool_calls:
            if isinstance(call, dict):
                tool_name = call.get("function", {}).get("name")
                tool_args_str = call.get("function", {}).get("arguments", "{}")
                call_id = call.get("id")
            else:
                tool_name = call.function.name
                tool_args_str = call.function.arguments
                call_id = call.id
            
            mock_data_str = await self._generate_mock_observation_with_llm(tool_name, tool_args_str)
            
            # 创建消息时使用正确的角色，agentscope 只接受 user, assistant, system
            resp_msg = Msg(
                name=self.name,
                role="assistant", 
                content=mock_data_str,
                metadata={"tool_call_id": call_id}
            )
            
            response_msgs.append(resp_msg)

        if len(response_msgs) == 1:
            return response_msgs[0]
        return response_msgs

    async def _generate_mock_observation_with_llm(self, tool_name: str, args_str: str) -> str:
        logger.info(f"LLM Simulator generating for: {tool_name} args={args_str}")

        prompt_content = (
            f"Please generate a JSON response for the tool: '{tool_name}'.\n"
            f"Input arguments: {args_str}\n"
            f"Ensure the response is consistent with the User Intent and Final State provided in the system prompt.\n"
            f"Return ONLY valid JSON."
        )

        messages = [
            {"role": "system", "content": self.base_sys_prompt},
            {"role": "user", "content": prompt_content}
        ]

        try:
            response = await self.model(messages=messages)
            
            content = ""
            for block in response.content:
                # 字典访问兼容
                block_type = block.get("type") if isinstance(block, dict) else block.type
                if block_type == "text":
                    text_val = block.get("text", "") if isinstance(block, dict) else block.text
                    content += text_val
            
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end != -1:
                json_str = content[start:end]
                json.loads(json_str) 
                return json_str
            else:
                logger.warning(f"Simulator LLM output invalid JSON: {content}")
                return json.dumps({"status": "error", "message": "Simulator generation format error"})

        except Exception as e:
            logger.error(f"Simulator LLM failed: {e}")
            return json.dumps({"status": "error", "message": "Simulation internal error"})