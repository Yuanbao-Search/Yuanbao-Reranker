# Evaluation

The evaluator reranks Top-100 first-stage candidates and reports nDCG@10 across the five benchmark panels. The complete task, subset, split, and dataset revision list is in [`benchmarks.json`](../src/yuanbao_reranker/benchmarks.json).

| Panel | Tasks | Subsets / Splits | Aggregation |
| :--- | ---: | ---: | :--- |
| CMTEB-R | 8 | 8 | Mean of 8 Chinese retrieval tasks |
| MMTEB-R | 16 | 49 | Subset means, then equal mean of 16 tasks |
| BEIR | 15 | 26 | Combine 12 CQADupStack subsets, then mean of 15 tasks |
| MLDR | 1 | 13 | Equal mean of 13 language subsets |
| MTEB-Code | 12 | 39 | Subset means, then equal mean of 12 programming tasks |

Each evaluated subset uses `ndcg_at_10`, rounded to five decimal places before panel aggregation. Avg. is the equal mean of the five panel scores.

## Data Preparation

The input consists of four files:

| File | Format |
| :--- | :--- |
| `corpus.jsonl` | One `{"id": "document-id", "text": "document text"}` per line |
| `queries.jsonl` | One `{"id": "query-id", "text": "query text"}` per line |
| `qrels.json` | `{"query-id": {"document-id": relevance_grade}}` |
| `candidates.json` | `{"query-id": {"document-id": retrieval_score}}` |

Keep all relevance labels, including relevant documents outside the retrieved candidates. Use the same corpus and query export that produced the retrieval results.

MTEB data can be exported in a separate environment:

```bash
python -m pip install -r requirements/data.txt
python -m pip install -e .
python -m yuanbao_reranker.export_mteb \
  --benchmark CMTEB-R --task T2Retrieval --subset default --split dev \
  --output data/T2Retrieval
```

The exported `dataset.json` records whether identical query and document IDs are excluded from metrics.

## First-Stage Retrieval

Install `requirements/retrieve.txt` to generate candidates with Qwen3-Embedding-0.6B:

```bash
python -m yuanbao_reranker.retrieve \
  --task T2Retrieval \
  --corpus data/T2Retrieval/corpus.jsonl \
  --queries data/T2Retrieval/queries.jsonl \
  --output data/T2Retrieval/candidates.json
```

The retriever uses normalized last-token embeddings and cosine similarity. Reuse one candidate file across rerankers for a matched comparison.

## Reranking

In the evaluation environment, prepare a task subset:

```bash
python -m yuanbao_reranker.prepare \
  --benchmark CMTEB-R --task T2Retrieval --subset default --split dev \
  --corpus data/T2Retrieval/corpus.jsonl \
  --queries data/T2Retrieval/queries.jsonl \
  --qrels data/T2Retrieval/qrels.json \
  --candidates data/T2Retrieval/candidates.json \
  --output data/prepared/T2Retrieval

python -m yuanbao_reranker.evaluate \
  --model Yuanbao-Search/Yuanbao-Reranker-0.6B \
  --config configs/eval-0.6B.json \
  --data data/prepared/T2Retrieval \
  --output results/0.6B/T2Retrieval
```

`prepare` reads the identical-ID policy from `dataset.json` beside the query file. For a manual export without that file, pass `--ignore-identical-ids` or `--no-ignore-identical-ids` according to the benchmark. Use separate output directories for each task, subset, and split.

Use the corresponding `eval-4B.json` or `eval-8B.json` for the larger models. The configurations select FP16, a 10,000-token engine limit, prefix caching, and model-specific batching:

| Model | Panel | Tensor Parallel Size | Batch Size |
| :--- | :--- | ---: | ---: |
| 0.6B | Text panels | 8 | 16 |
| 0.6B | MTEB-Code | 1 | 512 |
| 4B | All | 1 | 128 |
| 8B | All | 1 | 64 |

To run 0.6B evaluation on a single GPU instead of an 8-GPU node, pass `--tensor-parallel-size 1 --distributed-executor-backend uni`.

## Prompts and Scoring

The encoded system/user input is truncated to 8,192 tokens before the assistant suffix is appended. Relevance is computed from the `yes` and `no` token log probabilities:

$$s = \frac{\exp(\ell_{\mathrm{yes}})}{\exp(\ell_{\mathrm{yes}})+\exp(\ell_{\mathrm{no}})}.$$

## Aggregate

```bash
python -m yuanbao_reranker.aggregate \
  --results results/0.6B \
  --output results/0.6B-summary.json
```

The command verifies that all benchmark tasks and subsets in the inventory are present, and writes the panel scores and Avg. on the 0–100 scale.
