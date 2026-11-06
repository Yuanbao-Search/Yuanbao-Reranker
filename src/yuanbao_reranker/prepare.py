import argparse
import json
import math
from pathlib import Path

from .benchmark import find_unit, prompts


def read_jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def prepare(unit, corpus_path, queries_path, qrels_path, candidates_path, output, ignore_identical_ids=None):
    source_metadata = Path(queries_path).parent / "dataset.json"
    if source_metadata.exists():
        source = json.loads(source_metadata.read_text())
        if any(source.get(key) != unit[key] for key in unit):
            raise ValueError("Dataset metadata does not match the requested unit")
        if ignore_identical_ids is not None and ignore_identical_ids != source["ignore_identical_ids"]:
            raise ValueError("Identical-ID policy differs from dataset metadata")
        ignore_identical_ids = source["ignore_identical_ids"]
    elif ignore_identical_ids is None:
        raise ValueError("Provide dataset.json or an explicit identical-ID policy")
    candidates = json.loads(Path(candidates_path).read_text())
    qrels = json.loads(Path(qrels_path).read_text())
    queries = {}
    for row in read_jsonl(queries_path):
        qid = str(row["id"])
        if qid in queries or not isinstance(row["text"], str):
            raise ValueError("Duplicate query ID or non-text query")
        queries[qid] = row["text"]
    if not queries or set(queries) != set(qrels):
        raise ValueError("Queries and qrels must have identical nonempty query sets")
    selected = {}
    for qid in queries:
        if qid not in candidates or not candidates[qid]:
            raise ValueError(f"Missing candidates for {qid}")
        scores = candidates[qid]
        if any(not math.isfinite(float(value)) for value in scores.values()):
            raise ValueError("Candidate scores must be finite")
        selected[qid] = sorted(scores, key=lambda did: float(scores[did]), reverse=True)[:100]
    needed = {did for ids in selected.values() for did in ids}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seen = set()
    with (output / "documents.jsonl").open("w") as handle:
        for row in read_jsonl(corpus_path):
            did = str(row["id"])
            if did in needed:
                if did in seen or not isinstance(row["text"], str):
                    raise ValueError("Duplicate document ID or non-text document")
                seen.add(did)
                handle.write(json.dumps({"id": did, "text": row["text"]}, ensure_ascii=False) + "\n")
    if seen != needed:
        raise ValueError(f"Missing {len(needed - seen)} candidate documents")
    with (output / "queries.jsonl").open("w") as handle:
        for qid, query in queries.items():
            handle.write(json.dumps({"id": qid, "text": query, "documents": selected[qid]}, ensure_ascii=False) + "\n")
    (output / "qrels.json").write_text(json.dumps(qrels, ensure_ascii=False))
    metadata = dict(unit, instruction=prompts()[unit["task"]], top_k=100,
                    ignore_identical_ids=ignore_identical_ids, queries=len(queries))
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    for name in ("benchmark", "task", "subset", "split", "corpus", "queries", "qrels", "candidates", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--ignore-identical-ids", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    unit = find_unit(args.benchmark, args.task, args.subset, args.split)
    prepare(unit, args.corpus, args.queries, args.qrels, args.candidates, args.output, args.ignore_identical_ids)


if __name__ == "__main__":
    main()
