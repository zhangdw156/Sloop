from sloop.core import GraphBuilder
from sloop.utils import setup_logging

setup_logging()

jsonl_path = "/dfs/data/work/Sloop/data/tools.jsonl"

builder = GraphBuilder()
builder.load_from_jsonl(jsonl_path)
builder.build()
builder.save_checkpoint()
# builder.load_checkpoint("/dfs/data/work/Sloop/data/graph_checkpoint.pkl")
builder.export_graph_json()
builder.export_graphml()
builder.show()
