import json
from importlib.resources import files


def inventory():
    return json.loads(files("yuanbao_reranker").joinpath("benchmarks.json").read_text())


def prompts():
    return json.loads(files("yuanbao_reranker").joinpath("task_prompts.json").read_text())


def unit_key(unit):
    return (unit["benchmark"], unit["task"], unit["subset"], unit["split"])


def find_unit(benchmark, task, subset, split):
    key = (benchmark, task, subset, split)
    matches = [unit for unit in inventory() if unit_key(unit) == key]
    if len(matches) != 1:
        raise ValueError(f"Unknown benchmark unit: {key}")
    return matches[0]
