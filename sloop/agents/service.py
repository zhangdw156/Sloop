"""
服务模拟器 (Service Agent)

模拟API服务执行，根据工具调用更新环境状态。
"""

import logging
import json
from typing import Dict, Any, Optional
from sloop.models import ToolCall, Blueprint, EnvState
from sloop.utils.llm import chat_completion
from sloop.utils.template import render_service_prompt

logger = logging.getLogger(__name__)


class ServiceAgent:
    """
    服务智能体

    负责模拟API服务调用，根据工具调用和当前状态生成合理的响应，
    并更新环境状态。
    """

    def __init__(self):
        """初始化服务智能体"""
        logger.info("ServiceAgent initialized")

    def execute_tool(
        self,
        tool_call: ToolCall,
        current_state: EnvState,
        blueprint: Blueprint
    ) -> Dict[str, Any]:
        """
        执行工具调用

        参数:
            tool_call: 工具调用信息
            current_state: 当前环境状态
            blueprint: 对话蓝图（用于参考）

        返回:
            包含响应和状态更新的字典
        """
        logger.info(f"Executing tool: {tool_call.name}")

        # 构造提示
        prompt = render_service_prompt(tool_call, current_state, blueprint)

        # 调用LLM生成服务响应
        response = chat_completion(
            prompt=prompt,
            system_message="You are an API simulator. Generate realistic responses and state updates based on the tool call.",
            json_mode=True
        )

        if not response or response.startswith("调用错误"):
            logger.error(f"Failed to execute tool: {response}")
            return {
                "response": f"Error executing {tool_call.name}",
                "state_updates": {}
            }

        try:
            # 解析LLM响应
            result = json.loads(response)
            logger.info(f"Tool execution successful: {tool_call.name}")

            # 验证响应格式
            if not isinstance(result, dict):
                raise ValueError("Response must be a dictionary")

            if "response" not in result:
                result["response"] = f"Executed {tool_call.name}"

            if "state_updates" not in result:
                result["state_updates"] = {}

            # 确保state_updates是字典
            if not isinstance(result["state_updates"], dict):
                result["state_updates"] = {}

            return result

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse service response: {e}")
            return {
                "response": f"Executed {tool_call.name} (response parsing failed)",
                "state_updates": {}
            }

    def update_state(self, current_state: EnvState, state_updates: Dict[str, Any]) -> EnvState:
        """
        更新环境状态

        参数:
            current_state: 当前状态
            state_updates: 状态更新字典

        返回:
            更新后的新状态
        """
        # 创建状态副本
        new_state = current_state.model_copy()

        # 应用更新到状态字典
        new_state.update(state_updates)

        logger.info(f"State updated with {len(state_updates)} changes")
        return new_state


# ==================== 测试代码 ====================

if __name__ == "__main__":
    logger.info("🔧 Service Agent 测试")
    logger.info("=" * 50)

    from sloop.models import ToolCall, Blueprint, EnvState

    # 创建模拟工具调用
    mock_tool_call = ToolCall(
        tool_name="search_restaurants",
        arguments={"city": "Shanghai", "cuisine": "Italian"}
    )

    # 创建模拟状态
    mock_state = EnvState(
        state={
            "restaurant_found": False,
            "menu_loaded": False,
            "booking_confirmed": False
        }
    )

    # 创建模拟blueprint
    mock_blueprint = Blueprint(
        intent="查找餐厅并预订",
        required_tools=["search_restaurants", "book_restaurant"],
        ground_truth=["search_restaurants", "book_restaurant"],
        initial_state={"restaurant_found": False, "booking_confirmed": False},
        expected_state={"restaurant_found": True, "booking_confirmed": True}
    )

    logger.info("📋 测试数据:")
    logger.info(f"  工具调用: {mock_tool_call.name}")
    logger.info(f"  参数: {mock_tool_call.arguments}")
    logger.info(f"  当前状态: {mock_state.model_dump()}")
    logger.info("")

    # 初始化服务智能体
    logger.info("🔧 初始化ServiceAgent...")
    service_agent = ServiceAgent()

    logger.info("⚙️ 执行工具调用...")
    try:
        result = service_agent.execute_tool(mock_tool_call, mock_state, mock_blueprint)

        logger.info("✅ 执行成功！")
        logger.info(f"📝 响应: {result['response']}")
        logger.info(f"🔄 状态更新: {result['state_updates']}")

        # 应用状态更新
        updated_state = service_agent.update_state(mock_state, result['state_updates'])
        logger.info(f"📊 更新后状态: {updated_state.model_dump()}")

    except Exception as e:
        logger.error(f"❌ 执行失败: {e}")

        # 如果LLM调用失败，提供模拟结果
        logger.info("\n🔧 提供模拟服务响应:")
        mock_result = {
            "response": "Found 5 Italian restaurants in Shanghai",
            "state_updates": {"restaurant_found": True}
        }
        logger.info(f"响应: {mock_result['response']}")
        logger.info(f"状态更新: {mock_result['state_updates']}")

        # 应用模拟更新
        updated_state = service_agent.update_state(mock_state, mock_result['state_updates'])
        logger.info(f"更新后状态: {updated_state.model_dump()}")

    logger.info("\n✅ Service Agent 测试完成！")
