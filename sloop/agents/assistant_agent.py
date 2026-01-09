import json
from typing import Union, Sequence, List, Dict
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel

from sloop.prompts.simulation import ASSISTANT_SYSTEM_PROMPT
from sloop.configs.env import env_config

class AssistantAgent(AgentBase):
    def __init__(
        self, 
        name: str, 
        tools_list: List[Dict], 
        **kwargs
    ):
        super().__init__()
        
        self.name = name
        self.memory = InMemoryMemory()
        
        # 预处理 Tools 格式
        self.openai_tools = self._format_tools_for_openai(tools_list)
        
        # 提取工具名称用于 System Prompt
        tools_names = []
        for t in tools_list:
            if "function" in t:
                tools_names.append(t["function"].get("name", "unknown"))
            else:
                tools_names.append(t.get("name", "unknown"))
                
        tools_desc_str = ", ".join(tools_names)

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
                "temperature": 0.1,
                "max_tokens": 4096
            },
            stream=False
        )

        sys_content = ASSISTANT_SYSTEM_PROMPT.format(tools_desc=tools_desc_str)
        self.sys_prompt_dict = {"role": "system", "content": sys_content}

    def _format_tools_for_openai(self, tools_list: List[Dict]) -> List[Dict]:
        formatted_tools = []
        for tool in tools_list:
            if "type" in tool and "function" in tool:
                formatted_tools.append(tool)
            else:
                formatted_tools.append({
                    "type": "function",
                    "function": tool
                })
        return formatted_tools

    @override
    async def reply(self, x: Union[Msg, Sequence[Msg]] = None) -> Msg:
        if x:
            if isinstance(x, list): await self.memory.add(x)
            else: await self.memory.add(x)

        history_msgs = await self.memory.get_memory()
        
        openai_messages = [self.sys_prompt_dict]
        
        for m in history_msgs:
            msg_dict = {"role": m.role, "content": str(m.content or "")}
            
            # 使用 getattr 安全获取动态属性
            tool_calls = getattr(m, "tool_calls", None)
            if tool_calls:
                 msg_dict["tool_calls"] = tool_calls
            
            if m.role == "tool":
                # 使用 getattr 安全获取 tool_call_id
                t_id = getattr(m, "tool_call_id", None)
                if t_id:
                    msg_dict["tool_call_id"] = t_id
            
            if m.name: msg_dict["name"] = m.name
            openai_messages.append(msg_dict)

        response = await self.model(
            messages=openai_messages, 
            tools=self.openai_tools
        )
        
        text_content = ""
        tool_calls_list = []

        for block in response.content:
            # 字典访问修复
            block_type = block.get("type") if isinstance(block, dict) else block.type
            
            if block_type == "text":
                text_val = block.get("text", "") if isinstance(block, dict) else block.text
                text_content += text_val
            
            elif block_type == "tool_use":
                if isinstance(block, dict):
                    b_id = block.get("id")
                    b_name = block.get("name")
                    b_input = block.get("input")
                else:
                    b_id = block.id
                    b_name = block.name
                    b_input = block.input

                tool_calls_list.append({
                    "id": b_id,
                    "type": "function",
                    "function": {
                        "name": b_name,
                        "arguments": json.dumps(b_input, ensure_ascii=False)
                    }
                })

        # [核心修复] 不在 __init__ 中传 tool_calls，而是手动绑定属性
        msg = Msg(
            name=self.name,
            role="assistant",
            content=text_content
        )
        
        if tool_calls_list:
            msg.tool_calls = tool_calls_list # 动态绑定属性
            # 为了兼容性，也可以试试作为 key (如果 Msg 继承自 dict)
            # msg["tool_calls"] = tool_calls_list 

        await self.memory.add(msg)
        return msg