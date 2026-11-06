import argparse
import json
import os

from yuanbao_reranker import YuanbaoReranker
from yuanbao_reranker.evaluate import evaluation_settings

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
args = parser.parse_args()
settings, environment = evaluation_settings(None, None)
os.environ.update(environment)
batch_size = settings.pop("batch_size")
model = YuanbaoReranker(args.model, backend=args.backend, **settings)
print(json.dumps(model.rerank(
    "What is the capital of China?",
    ["Beijing is the capital of China.", "Shanghai is a major financial center."],
    batch_size=batch_size,
), ensure_ascii=False, indent=2))
