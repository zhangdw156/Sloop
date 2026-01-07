import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# 导入你的类
from sloop.utils.tool_graph import ToolGraphBuilder


class TestToolGraphBuilder(unittest.TestCase):

    def setup(self):
        """测试准备：创建临时的 JSONL 数据"""
        self.test_data = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather info. Output contains city info.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "book_hotel",
                    "description": "Book a hotel.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name where hotel is located"}
                        },
                        "required": ["city"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "irrelevant_tool",
                    "description": "Do something completely different.",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]

    # @patch 装饰器用于拦截 ToolGraphBuilder 中导入的 RemoteEmbeddingService
    @patch('sloop.utils.tool_graph.RemoteEmbeddingService')
    def test_build_graph_with_mock(self, MockServiceClass):
        """
        测试构建图谱（Mock 模式，无需真实调用 API）
        """
        # 1. 配置 Mock 对象
        mock_service_instance = MockServiceClass.return_value
        mock_service_instance.model_name = "mock-bge-m3"

        # 模拟 get_embedding：不管输入什么，都返回一个固定长度的假向量
        # 假设向量维度为 3
        mock_service_instance.get_embedding.return_value = [0.1, 0.2, 0.3]

        # 模拟 compute_similarity：自定义相似度逻辑
        # 我们在这里通过判断输入向量是否相等，或者简单地根据调用次数来造假
        # 但更简单的方法是：直接控制逻辑。

        def side_effect_similarity(vec_a, vec_b):
            # 这里是一个 hack：我们在测试数据里知道
            # get_weather (Producer) 和 book_hotel (Consumer) 应该产生关联
            # 在这里我们简单粗暴地返回高相似度，以此验证 graph.add_edge 逻辑是否通畅
            return 0.95  # 强制认为所有对比都非常相似 (> 0.75 阈值)

        mock_service_instance.compute_similarity.side_effect = side_effect_similarity

        # 2. 创建临时文件并运行测试
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test_tools.jsonl"
            with open(test_file, 'w', encoding='utf-8') as f:
                for item in self.test_data:
                    f.write(json.dumps(item) + '\n')

            # 初始化 Builder
            builder = ToolGraphBuilder()

            # 验证 Mock 服务是否被注入
            assert builder.embedding_service is not None
            print(f"Service used in test: {builder.embedding_service}") # 应该显示是个 Mock 对象

            # 加载数据
            builder.load_from_jsonl(str(test_file))
            assert len(builder.tools) == 3

            # 3. 构建图谱
            # 因为我们的 side_effect_similarity 总是返回 0.95
            # 所以理论上所有两两组合（除了自己）都会建立边
            graph = builder.build(similarity_threshold=0.8)

            # 4. 验证核心逻辑
            # 检查是否有节点
            assert graph.number_of_nodes() == 3

            # 检查是否生成了边 (由于我们强制返回高相似度，这里肯定有边)
            assert graph.has_edge("get_weather", "book_hotel")

            # 打印生成的边，看看是否包含了我们期望的属性
            edge_data = graph.get_edge_data("get_weather", "book_hotel")
            # 注意：MultiDiGraph 返回的是一个字典，key 是边的 index
            print("Edge Data:", edge_data)

            # 验证边属性是否存在
            first_edge = list(edge_data.values())[0]
            assert "weight" in first_edge
            assert first_edge["weight"] == 0.95
            assert first_edge["relation"] == "provides_parameter"

            # 5. 测试导出和可视化不会报错
            output_img = Path(tmpdir) / "graph.png"
            builder.visualize(str(output_img))
            assert output_img.exists()

if __name__ == "__main__":
    unittest.main()
