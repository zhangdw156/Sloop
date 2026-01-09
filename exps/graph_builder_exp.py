from sloop.core import GraphBuilder
from sloop.utils import setup_logging

setup_logging()

builder = GraphBuilder()
builder.load_from_jsonl("/root/work/Sloop/tests/data/tools.jsonl")
builder.build()
builder.save_checkpoint()
# builder.load_checkpoint("/dfs/data/work/Sloop/data/graph_checkpoint.pkl")
# builder.visualize()
# builder.prune_graph()
builder.export_graph_json()
builder.export_graphml()
builder.show()
