import math


def ndcg_at_10(qrels, predictions, ignore_identical_ids=False):
    import pytrec_eval
    if not qrels:
        raise ValueError("qrels must not be empty")
    for qid, docs in predictions.items():
        if qid not in qrels or any(not math.isfinite(float(v)) for v in docs.values()):
            raise ValueError("Invalid prediction")
    run = {qid: {did: float(score) for did, score in docs.items()
                if not ignore_identical_ids or qid != did}
           for qid, docs in predictions.items()}
    graded = {qid: {did: int(value) for did, value in docs.items()} for qid, docs in qrels.items()}
    for qid, docs in qrels.items():
        if any(float(value) != graded[qid][did] or value < 0 for did, value in docs.items()):
            raise ValueError("qrels must contain nonnegative integer relevance grades")
    evaluator = pytrec_eval.RelevanceEvaluator(graded, {"ndcg_cut.10"})
    scores = evaluator.evaluate({qid: docs for qid, docs in run.items() if docs})
    return round(sum(scores.get(qid, {}).get("ndcg_cut_10", 0.0) for qid in graded) / len(graded), 5)
