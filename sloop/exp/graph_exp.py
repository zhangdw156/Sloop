from sloop.utils import ToolGraphBuilder, setup_logging

setup_logging()

graph = ToolGraphBuilder()
# graph.load_from_jsonl("/root/work/Sloop/tests/data/tools.jsonl")
# graph.build()
# graph.visualize()
graph.load_checkpoint("/dfs/data/work/Sloop/data/graph_checkpoint.pkl")
# graph.prune_graph()
# graph.export_graph_json()
# graph.export_graphml()
# graph.save_checkpoint()
graph.show()
