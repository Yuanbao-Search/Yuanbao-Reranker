import argparse
import json
import os
from pathlib import Path

from .benchmark import find_unit, prompts
from .metrics import ndcg_at_10
from .model import YuanbaoReranker
from .prepare import read_jsonl


def evaluation_settings(path, benchmark, overrides=None):
    settings = dict(batch_size=32, tensor_parallel_size=1, dtype="float16", max_model_len=10000,
                    max_num_seqs=512, gpu_memory_utilization=0.8, enable_prefix_caching=True,
                    distributed_executor_backend="uni")
    allowed = set(settings)
    environment = {"VLLM_USE_V1": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "0"}
    if path:
        config = json.loads(Path(path).read_text())
        settings.update(config["defaults"])
        settings.update(config.get("benchmarks", {}).get(benchmark, {}))
        environment.update(config.get("environment", {}))
    settings.update({key: value for key, value in (overrides or {}).items() if value is not None})
    if set(settings) != allowed or set(environment) != {"VLLM_USE_V1", "VLLM_ENABLE_V1_MULTIPROCESSING"}:
        raise ValueError("Unknown evaluation setting")
    return settings, environment


def evaluate(model, data, output, model_name, batch_size=32):
    data, output = Path(data), Path(output)
    metadata = json.loads((data / "metadata.json").read_text())
    unit = find_unit(*(metadata[k] for k in ("benchmark", "task", "subset", "split")))
    if metadata["dataset_revision"] != unit["dataset_revision"]:
        raise ValueError("Dataset revision differs from the benchmark configuration")
    if metadata["instruction"] != prompts()[unit["task"]] or metadata["top_k"] != 100:
        raise ValueError("Unexpected instruction or candidate count setting")
    docs = {row["id"]: row["text"] for row in read_jsonl(data / "documents.jsonl")}
    qrels = json.loads((data / "qrels.json").read_text())
    predictions = {}
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    batch = []
    def flush():
        scores = model.score([(query, docs[did]) for qid, query, did in batch], metadata["instruction"], batch_size)
        if len(scores) != len(batch):
            raise ValueError("Incomplete scores")
        for (qid, query, did), score in zip(batch, scores):
            predictions[qid][did] = score
        batch.clear()
    for row in read_jsonl(data / "queries.jsonl"):
        qid, ids = row["id"], row["documents"]
        if qid in predictions or not 0 < len(ids) <= 100 or len(ids) != len(set(ids)):
            raise ValueError("Invalid query or candidate list")
        predictions[qid] = {}
        for did in ids:
            batch.append((qid, row["text"], did))
            if len(batch) == batch_size:
                flush()
    if batch:
        flush()
    if set(predictions) != set(qrels) or len(predictions) != metadata["queries"]:
        raise ValueError("Incomplete query coverage")
    score = ndcg_at_10(qrels, predictions, metadata["ignore_identical_ids"])
    result = dict(unit, model=model_name, ndcg_at_10=score, queries=len(predictions),
                  instruction=metadata["instruction"], max_length=model.max_length, top_k=100,
                  batch_size=batch_size, runtime=getattr(model, "runtime", {}))
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions.json").write_text(json.dumps(predictions, allow_nan=False))
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"))
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--distributed-executor-backend", choices=("uni", "mp", "ray"))
    args = parser.parse_args()
    metadata = json.loads((Path(args.data) / "metadata.json").read_text())
    overrides = {key: value for key, value in vars(args).items() if key not in {"model", "data", "output", "config"}}
    settings, environment = evaluation_settings(args.config, metadata["benchmark"], overrides)
    os.environ.update(environment)
    batch_size = settings.pop("batch_size")
    model = YuanbaoReranker(args.model, **settings)
    print(json.dumps(evaluate(model, args.data, args.output, args.model, batch_size), indent=2))


if __name__ == "__main__":
    main()
