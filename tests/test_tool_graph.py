import os
import json
import tempfile
from pathlib import Path
from sloop.utils.tool_graph import ToolGraphBuilder

def test_tool_graph_builder():
    # 创建临时的测试数据
    # 根据 tests/data/tools.jsonl 的实际格式，每行是一个独立的工具定义
    test_data = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a city. Returns temperature and conditions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["city"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "book_hotel",
                "description": "Book a hotel room. Requires city, check-in date, and number of guests.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "check_in": {"type": "string", "format": "date"},
                        "guests": {"type": "integer"}
                    },
                    "required": ["city", "check_in", "guests"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "book_flight",
                "description": "Book a flight. Requires departure city, destination city, and date.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "string", "description": "Departure city"},
                        "to": {"type": "string", "description": "Destination city"},
                        "date": {"type": "string", "format": "date"}
                    },
                    "required": ["from", "to", "date"]
                }
            }
        }
    ]

    # 在临时目录中创建测试文件
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_tools.jsonl"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        # 初始化构建器并加载数据
        builder = ToolGraphBuilder()
        builder.load_from_jsonl(str(test_file))
        
        # 验证加载的工具数量
        assert len(builder.tools) == 3, f"Expected 3 tools, got {len(builder.tools)}"
        assert "get_weather" in builder.tools
        assert "book_hotel" in builder.tools
        assert "book_flight" in builder.tools

        # 构建图谱
        graph = builder.build()
        
        # 验证图谱结构
        assert graph.number_of_nodes() == 3, f"Expected 3 nodes, got {graph.number_of_nodes()}"
        
        # 检查依赖关系：get_weather 描述中包含 'city'，book_hotel 需要 'city'，应存在依赖
        assert graph.has_edge("get_weather", "book_hotel"), "Expected edge from get_weather to book_hotel"
        
        # 检查其他可能的依赖
        # book_flight 需要 'from', 'to', 'date'，没有工具的描述中包含这些，不应有边指向它
        # 但 book_hotel 需要 'check_in'，没有工具提供，所以没有入边
        # get_weather 没有必需参数被其他工具满足，所以没有入边

        # 测试可视化功能
        output_image = Path(tmpdir) / "test_graph.png"
        builder.visualize(str(output_image))
        assert output_image.exists(), "Visualization file was not created"
        
        # 测试导出功能
        output_json = Path(tmpdir) / "test_graph.json"
        builder.export_graph_json(str(output_json))
        assert output_json.exists(), "Exported JSON file was not created"
        
        print("All tests passed successfully!")

if __name__ == "__main__":
    test_tool_graph_builder()
