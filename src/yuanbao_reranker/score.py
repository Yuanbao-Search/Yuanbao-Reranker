import argparse
import json
import os

from .benchmark import inventory, prompts
from .evaluate import evaluation_settings
from .model import YuanbaoReranker
from .prompts import DEFAULT_INSTRUCTION


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--doc", action="append", required=True)
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--config")
    parser.add_argument("--benchmark", choices=sorted({u["benchmark"] for u in inventory()}))
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--task", choices=sorted(prompts()))
    prompt.add_argument("--instruction")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--distributed-executor-backend", choices=("uni", "mp", "ray"))
    args = parser.parse_args(argv)
    benchmark = args.benchmark
    instruction = DEFAULT_INSTRUCTION if args.instruction is None else args.instruction
    if args.task:
        panels = {u["benchmark"] for u in inventory() if u["task"] == args.task}
        if benchmark and benchmark not in panels:
            parser.error("Task does not belong to the selected benchmark")
        if benchmark is None:
            if len(panels) != 1:
                parser.error("Specify --benchmark for this task")
            benchmark = next(iter(panels))
        instruction = prompts()[args.task]
    if args.config and benchmark is None:
        parser.error("--config requires --task or --benchmark")
    if args.device and args.backend != "transformers":
        parser.error("--device is only supported by the transformers backend")
    if args.dtype == "float32" and args.backend != "transformers":
        parser.error("Use float16 or bfloat16 for the evaluation vLLM backend")
    excluded = {"model", "query", "doc", "backend", "config", "benchmark", "task", "instruction", "device"}
    overrides = {k: v for k, v in vars(args).items() if k not in excluded}
    settings, environment = evaluation_settings(args.config, benchmark, overrides)
    os.environ.update(environment)
    batch_size = settings.pop("batch_size")
    if batch_size < 1:
        parser.error("--batch-size must be positive")
    extra = {} if args.backend == "vllm" else {"backend": "transformers", "device": args.device}
    model = YuanbaoReranker(args.model, **settings, **extra)
    scores = model.score([(args.query, document) for document in args.doc], instruction, batch_size)
    result = {
        "model": args.model, "backend": args.backend, "task": args.task,
        "benchmark": benchmark, "instruction": instruction, "query": args.query,
        "max_length": model.max_length, "batch_size": batch_size, "runtime": model.runtime,
        "documents": [{"document": d, "relevance_score": s} for d, s in zip(args.doc, scores)],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
