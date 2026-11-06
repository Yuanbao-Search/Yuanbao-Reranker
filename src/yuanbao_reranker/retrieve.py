import argparse
import json
from importlib.resources import files
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .prepare import read_jsonl


def last_token_pool(hidden, mask):
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    positions = positions.masked_fill(mask == 0, -1).max(dim=1).values
    if (positions < 0).any():
        raise ValueError("Empty embedding input")
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    for name in ("corpus", "queries", "task", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--query-block-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1 or args.query_block_size < 1:
        raise ValueError("Batch sizes must be positive")
    instruction = json.loads(files("yuanbao_reranker").joinpath("retrieval_prompts.json").read_text())[args.task]
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", trust_remote_code=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=False,
    ).to(device).eval()
    @torch.inference_mode()
    def encode(texts):
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=8192, return_tensors="pt").to(device)
        hidden = model(**encoded).last_hidden_state
        return F.normalize(last_token_pool(hidden, encoded["attention_mask"]).float(), p=2, dim=1).cpu()
    queries = list(read_jsonl(args.queries))
    if not queries or len({row["id"] for row in queries}) != len(queries):
        raise ValueError("Queries must have unique IDs")
    embeddings = torch.cat([encode([f'Instruct: {instruction}\nQuery: {r["text"]}'
                                   for r in queries[i:i + args.batch_size]])
                            for i in range(0, len(queries), args.batch_size)])
    best_scores = torch.empty((len(queries), 0))
    best_ids = [[] for _ in queries]
    seen = set()
    batch = []
    def update(rows):
        nonlocal best_scores, best_ids
        document_vectors = encode([row["text"] for row in rows])
        ids = [str(row["id"]) for row in rows]
        next_scores, next_ids = [], []
        for start in range(0, len(queries), args.query_block_size):
            end = min(start + args.query_block_size, len(queries))
            scores = embeddings[start:end] @ document_vectors.T
            combined = torch.cat((best_scores[start:end], scores), dim=1)
            order = torch.argsort(combined, dim=1, descending=True, stable=True)[:, :100]
            next_scores.append(combined.gather(1, order))
            for i, indices in enumerate(order.tolist(), start=start):
                combined_ids = best_ids[i] + ids
                next_ids.append([combined_ids[j] for j in indices])
        best_scores = torch.cat(next_scores)
        best_ids = next_ids
    for row in read_jsonl(args.corpus):
        did = str(row["id"])
        if did in seen:
            raise ValueError("Duplicate document ID")
        seen.add(did)
        batch.append(row)
        if len(batch) == args.batch_size:
            update(batch)
            batch = []
    if batch:
        update(batch)
    if not seen:
        raise ValueError("Corpus is empty")
    run = {str(query["id"]): dict(zip(ids, scores)) for query, ids, scores in zip(queries, best_ids, best_scores.tolist())}
    with Path(args.output).open("x") as handle:
        json.dump(run, handle, allow_nan=False)


if __name__ == "__main__":
    main()
