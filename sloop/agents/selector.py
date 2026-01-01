"""
裁判智能体 (Selector Agent)

负责在候选工具中选择最佳的下一个工具，或决定结束任务。
基于当前的工具链条和候选工具列表，使用 LLM 进行智能决策。
"""

from typing import List, Optional

from sloop.config import get_settings
from sloop.models import ToolDefinition
from sloop.utils.llm import chat_completion
from sloop.utils.template import render_selector_prompt


class SelectorAgent:
    """
    裁判智能体

    分析当前工具链条，选择最合适的下一个工具或决定结束任务。
    """

    def __init__(self):
        """
        初始化裁判智能体
        """
        self.settings = get_settings()

    def select_next_tool(
        self,
        current_chain: List[str],
        candidates: List[ToolDefinition]
    ) -> Optional[str]:
        """
        选择下一个工具

        参数:
            current_chain: 当前已执行的工具名称列表
            candidates: 候选工具定义列表

        返回:
            选中的工具名称，或 None 表示结束任务
        """
        if not candidates:
            return None

        # 使用模板渲染提示
        user_prompt = render_selector_prompt(current_chain, candidates)

        # 调用 LLM（系统提示已在模板中）
        response = chat_completion(
            prompt=user_prompt,
            system_message="",
            json_mode=False,
        )

        if not response or response.startswith("调用错误"):
            # LLM 调用失败，默认结束任务
            return None

        # 清理响应
        result = response.strip()

        # 检查是否选择结束
        if result.upper() == "FINISH":
            return None

        # 检查是否是有效的工具名称
        valid_names = {tool.name for tool in candidates}
        if result in valid_names:
            return result

        # 如果不是有效名称，尝试提取工具名称
        for tool in candidates:
            if tool.name in result:
                return tool.name

        # 如果无法识别，默认结束任务
        return None


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("🎯 SelectorAgent 测试")
    print("=" * 50)

    # 创建模拟工具
    mock_candidates = [
        ToolDefinition(
            name="recommend_clothes",
            description="根据天气推荐穿衣",
            parameters={
                "type": "object",
                "properties": {
                    "weather": {"type": "string", "description": "天气情况"},
                },
                "required": ["weather"],
            },
        ),
        ToolDefinition(
            name="book_flight",
            description="预订机票",
            parameters={
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "目的地"},
                    "date": {"type": "string", "description": "出发日期"},
                },
                "required": ["destination"],
            },
        ),
        ToolDefinition(
            name="delete_database",
            description="删除数据库",
            parameters={
                "type": "object",
                "properties": {
                    "database_name": {"type": "string", "description": "数据库名称"},
                },
                "required": ["database_name"],
            },
        ),
    ]

    print(f"📋 候选工具: {len(mock_candidates)} 个")
    for tool in mock_candidates:
        print(f"  - {tool.name}: {tool.description}")

    # 初始化裁判智能体
    print("\n🤖 初始化 SelectorAgent...")
    selector = SelectorAgent()

    # 测试场景1: 已获取天气，推荐下一步
    print("\n🧪 测试场景1: 当前链条 ['get_weather']")
    current_chain1 = ["get_weather"]
    result1 = selector.select_next_tool(current_chain1, mock_candidates)
    print(f"🎯 选择结果: {result1}")

    # 测试场景2: 空链条
    print("\n🧪 测试场景2: 当前链条 []")
    current_chain2 = []
    result2 = selector.select_next_tool(current_chain2, mock_candidates)
    print(f"🎯 选择结果: {result2}")

    # 测试场景3: 已预订机票，可能结束
    print("\n🧪 测试场景3: 当前链条 ['book_flight']")
    current_chain3 = ["book_flight"]
    result3 = selector.select_next_tool(current_chain3, mock_candidates)
    print(f"🎯 选择结果: {result3}")

    print("\n✅ SelectorAgent 测试完成！")
