import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean

from .benchmark import inventory, prompts, unit_key


def aggregate(results):
    units = {unit_key(unit): unit for unit in inventory()}
    expected = set(units)
    found = {}
    for result in results:
        key = unit_key(result)
        if key in found:
            raise ValueError(f"Duplicate result: {key}")
        if key not in units or result["dataset_revision"] != units[key]["dataset_revision"]:
            raise ValueError("Unexpected dataset revision or unit")
        if result["instruction"] != prompts()[result["task"]] or result["max_length"] != 8192 or result["top_k"] != 100:
            raise ValueError("Results use different evaluation settings")
        value = float(result["ndcg_at_10"])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid nDCG@10: {key}")
        found[key] = value
    if set(found) != expected:
        raise ValueError(f"Incomplete benchmark: missing={len(expected - set(found))}, extra={len(set(found) - expected)}")
    if len({result["model"] for result in results}) != 1:
        raise ValueError("Results must belong to one model")
    tasks = defaultdict(lambda: defaultdict(list))
    for (panel, task, subset, split), value in found.items():
        if panel == "BEIR" and task.startswith("CQADupstack"):
            task = "CQADupStack"
        tasks[panel][task].append(value)
    panels = {panel: 100 * mean(mean(values) for values in grouped.values()) for panel, grouped in tasks.items()}
    return {"model": results[0]["model"], "panels": panels, "Avg.": mean(panels.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = aggregate([json.loads(path.read_text()) for path in sorted(Path(args.results).rglob("result.json"))])
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
