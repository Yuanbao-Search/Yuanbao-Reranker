import argparse
import json
from pathlib import Path

from .benchmark import find_unit


def main():
    parser = argparse.ArgumentParser()
    for name in ("benchmark", "task", "subset", "split", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    import mteb
    from mteb.evaluation.evaluators.RetrievalEvaluator import corpus_to_str
    from mteb.evaluation.evaluators.utils import convert_conv_history_to_query
    if mteb.__version__ != "1.38.9":
        raise RuntimeError("This benchmark requires mteb 1.38.9")
    unit = find_unit(args.benchmark, args.task, args.subset, args.split)
    task = mteb.get_tasks(tasks=[args.task])[0]
    if task.metadata.dataset["revision"] != unit["dataset_revision"]:
        raise ValueError("Unexpected dataset revision")
    if args.subset != "default":
        task.filter_languages(languages=None, hf_subsets=[args.subset])
    task.load_data(eval_splits=[args.split])
    def select(values):
        return values[args.split] if args.subset == "default" else values[args.subset][args.split]
    corpus, queries, qrels = (select(values) for values in (task.corpus, task.queries, task.relevant_docs))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "corpus.jsonl").open("w") as handle:
        for did, document in corpus.items():
            handle.write(json.dumps({"id": str(did), "text": corpus_to_str([document])[0]}, ensure_ascii=False) + "\n")
    with (output / "queries.jsonl").open("w") as handle:
        for qid, query in queries.items():
            if isinstance(query, list):
                query = convert_conv_history_to_query([query])[0]
            if not isinstance(query, str):
                raise ValueError(f"Unsupported query type: {type(query).__name__}")
            handle.write(json.dumps({"id": str(qid), "text": query}, ensure_ascii=False) + "\n")
    (output / "qrels.json").write_text(json.dumps(qrels, ensure_ascii=False))
    (output / "dataset.json").write_text(json.dumps(dict(unit, ignore_identical_ids=task.ignore_identical_ids), indent=2))


if __name__ == "__main__":
    main()
