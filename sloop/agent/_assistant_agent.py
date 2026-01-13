from typing import Any, Dict, List

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg, TextBlock, ToolResultBlock, ToolUseBlock
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit, ToolResponse  # 必须引入 ToolResponse

from .configs import env_config
from .prompts.simulation import ASSISTANT_SYSTEM_PROMPT


class AssistantAgent(ReActAgent):
    """
    AssistantAgent (ReAct Mode)

    Inherits from agentscope.agents.ReActAgent.
    Instead of executing local Python functions, it delegates the 'Acting' step
    to a SimulatorAgent to retrieve mock observations based on the Intent.
    """

    def __init__(
        self,
        name: str,
        tools_list: List[Dict],
        simulator: Any,
        max_iters: int = 10,
        verbose: bool = True,
        **kwargs,
    ):
        # 1. Initialize Model
        model_name = env_config.get("OPENAI_MODEL_NAME")
        base_url = env_config.get("OPENAI_MODEL_BASE_URL")
        api_key = env_config.get("OPENAI_MODEL_API_KEY") or "EMPTY"

        if not model_name or not base_url:
            raise ValueError("Missing model config in .env file!")

        model = OpenAIChatModel(
            model_name=model_name,
            api_key=api_key,
            client_kwargs={"base_url": base_url},
            generate_kwargs={
                "temperature": 0.1,
                "max_tokens": 4096,
            },
            stream=False,
        )

        # 2. Build Toolkit with Dummy Functions
        toolkit = Toolkit()
        self._register_dummy_tools(toolkit, tools_list)

        # 3. Prepare System Prompt
        sys_prompt = ASSISTANT_SYSTEM_PROMPT

        # 4. Initialize Parent ReActAgent
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model=model,
            toolkit=toolkit,
            formatter=OpenAIChatFormatter(),
            max_iters=max_iters,
            print_hint_msg=verbose,
            **kwargs,
        )

        # 5. Bind Simulator
        self.simulator = simulator

    def _register_dummy_tools(self, toolkit: Toolkit, tools_list: List[Dict]) -> None:
        """
        Register dummy functions into the toolkit using the provided JSON schemas.
        This ensures toolkit.get_json_schemas() returns the correct definitions for the LLM.
        """

        def create_dummy_func(tool_name):
            """Create a placeholder function that returns a valid ToolResponse."""

            def dummy_function(**kwargs):
                # [修复点] 这里必须返回 ToolResponse 对象
                return ToolResponse(
                    content=[
                        TextBlock(
                            type="text", text=f"Executed {tool_name} (Simulation)"
                        )
                    ]
                )

            # 设置函数名以便 AgentScope 识别
            dummy_function.__name__ = tool_name
            return dummy_function

        for tool_def in tools_list:
            # Extract function definition
            func_def = tool_def.get("function", tool_def)
            t_name = func_def.get("name")

            if not t_name:
                continue

            # Ensure complete schema structure
            if "function" in tool_def:
                full_schema = tool_def
            else:
                full_schema = {"type": "function", "function": func_def}

            # Register to Toolkit
            try:
                toolkit.register_tool_function(
                    tool_func=create_dummy_func(t_name),
                    json_schema=full_schema,
                    namesake_strategy="override",
                )
            except Exception as e:
                print(f"Warning: Failed to register dummy tool {t_name}: {e}")

    @override
    async def _acting(self, tool_call: ToolUseBlock) -> dict | None:
        """
        Override the acting process.
        """
        # 1. Construct the request message for the Simulator
        request_msg = Msg(
            name=self.name,
            role="assistant",
            content=[tool_call],  # Pass the ToolUseBlock directly in content
        )

        # 2. Call the Simulator
        sim_response_msg = await self.simulator.reply(request_msg)

        # 3. Extract output
        mock_result = sim_response_msg.get_text_content() or "{}"

        # 4. Create ToolResultBlock
        tool_res_block = ToolResultBlock(
            type="tool_result",
            id=tool_call["id"],
            name=tool_call["name"],
            output=mock_result,
        )

        tool_res_msg = Msg(
            name="system",
            role="system",
            content=[tool_res_block],
        )

        # 5. Add to Memory
        await self.memory.add(tool_res_msg)

        # 6. Return None (unless using structured output validation)
        return None
