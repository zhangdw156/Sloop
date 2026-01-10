from agentscope.message import Msg
from agentscope.agent import ReActAgent,UserAgent
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter
import asyncio
from sloop.configs.env import env_config

MODEL_NAME = env_config.get("OPENAI_MODEL_NAME") or "Qwen3-235B-A22B-Instruct-2507"
API_KEY = env_config.get("OPENAI_MODEL_API_KEY") or "EMPTY"
BASE_URL = env_config.get("OPENAI_MODEL_API_BASE")

react_agent=ReActAgent(
    name="Assistant",
    sys_prompt="你是一中科院软件所数科中心AI助手",
    model=OpenAIChatModel(
        model_name=MODEL_NAME,
        api_key=API_KEY,
        client_kwargs={"base_url":BASE_URL},
    ),
    formatter=OpenAIChatFormatter()
)

user_agent=UserAgent(
    name="User"      
)


async def main():
    msg=None
    while True:
        msg=await react_agent.reply(msg)
        print(msg)
        msg=await user_agent.reply(msg)
        print(msg)

if __name__ == "__main__":
    asyncio.run(main())
