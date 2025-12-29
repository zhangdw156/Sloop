"""
Service Agent 实现
负责模拟API工具的执行结果
"""

import json
from typing import List, Dict, Any


class ServiceAgent:
    """
    Service Agent 类
    模拟API工具的执行，生成Observation结果
    """

    def __init__(self, available_tools: List[Dict[str, Any]]):
        """
        初始化Service Agent

        Args:
            available_tools: 可用的工具列表，每个工具应包含 'name' 字段
        """
        # 将工具列表转换为字典，便于快速查找
        self.tools = {tool['name']: tool for tool in available_tools}

    def execute(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """
        执行工具调用

        Args:
            tool_name: 工具名称
            tool_args: 工具参数字典

        Returns:
            字符串类型的Observation结果
        """
        # 验证工具是否存在
        if tool_name not in self.tools:
            error_result = {
                "error": f"Tool '{tool_name}' not found in available tools"
            }
            return json.dumps(error_result, ensure_ascii=False)

        # Mock 执行逻辑（暂时返回模拟成功结果）
        mock_result = {
            "status": "success",
            "mock_data": f"Successfully executed {tool_name}",
            "tool_name": tool_name,
            "args": tool_args,
            "execution_time": "0.001s"  # 模拟执行时间
        }

        return json.dumps(mock_result, ensure_ascii=False)


# Self-Check 测试代码
if __name__ == "__main__":
    # 定义简单的测试工具Schema
    sample_tools = [
        {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "city": "string",
                "unit": "string"
            }
        }
    ]

    # 实例化ServiceAgent
    agent = ServiceAgent(sample_tools)

    # 测试用例1: 调用存在的工具
    print("=== 测试用例1: 调用存在的工具 ===")
    result1 = agent.execute("get_weather", {"city": "Beijing", "unit": "celsius"})
    print("结果:", result1)

    # 测试用例2: 调用不存在的工具
    print("\n=== 测试用例2: 调用不存在的工具 ===")
    result2 = agent.execute("non_existent_tool", {"param": "value"})
    print("结果:", result2)

    print("\n✅ Service Agent 测试完成")
