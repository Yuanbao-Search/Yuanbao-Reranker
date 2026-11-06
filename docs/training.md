# Training

Each JSONL row is one ranking group. Text and code use the same schema.

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `query` | string | Query text |
| `instruction` | string | Task description |
| `positive` | string | Relevant document |
| `negatives` | three strings | Negative documents |
| `teacher_scores` | four numbers | Teacher yes-logit minus no-logit, in positive/negative order |
| `eligible` | boolean, optional | Include the group in ranking loss; default `true` |

All four query–document pairs remain together in each training batch. `per_device_train_batch_size` counts groups; a value of 1 produces four model input sequences. Groups are shuffled with the configured data seed.

Use task-specific instructions, including code-search instructions. The benchmark instruction catalog is in [`task_prompts.json`](../src/yuanbao_reranker/task_prompts.json). Training data may supply additional task descriptions through `instruction`.

## Prompt

```text
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {instruction}
<Query>: {query}
<Document>: {document}<|im_end|>
<|im_start|>assistant
<think>

</think>

{yes_or_no}<|im_end|>
```

Training masks the system and user input and supervises the full assistant response, including the empty thinking block and closing token. The response is retained within the 4,096-token training limit; the source is truncated from the right.

## Objective

For each candidate, let $s_i = z_i^{\mathrm{yes}} - z_i^{\mathrm{no}}$ and let $t_i$ be the corresponding teacher score. For eligible groups, define $\mathcal{P}_g = \{(i,j): i < j,\ |t_i-t_j| > \epsilon\}$. For ineligible groups, set $\mathcal{P}_g = \varnothing$.

$$
\mathcal{L}_{\mathrm{rank}} = \frac{1}{G}\sum_g
\frac{\sum_{(i,j)\in\mathcal{P}_g}\max\left(0,\min(|t_i-t_j|,m)-\operatorname{sign}(t_i-t_j)(s_i-s_j)\right)}{\max(1,|\mathcal{P}_g|)},
\qquad
\mathcal{L} = \mathcal{L}_{\mathrm{SFT}} + \lambda\mathcal{L}_{\mathrm{rank}}.
$$

The defaults are $\lambda = 0.1$, $\epsilon = 0.2$, and $m = 4$. Groups marked ineligible, or whose teacher ranks a negative above the positive, contribute zero ranking loss and retain their SFT loss.

## Configuration

| Setting | Default |
| :--- | :--- |
| Training | Full-parameter fine-tuning |
| Sequence length | 4,096 |
| Learning rate | $10^{-5}$ |
| Schedule | Cosine, 2% warmup |
| Optimizer | AdamW, betas (0.9, 0.999), epsilon $10^{-8}$ |
| Weight decay | 0 |
| Gradient clipping | 1.0 |
| Precision | BF16 |
| Groups per device | 1 |
| Gradient accumulation | 1 |
| Distributed optimizer | DeepSpeed ZeRO-2 |

The configuration trains for one epoch on the supplied data. Set `max_steps` in `training` when a fixed update count is required. The original runs used 64 H20 GPUs; at that scale the effective batch contains 64 groups, or 256 sequences. With fewer GPUs, increase gradient accumulation to keep the same group batch size.

The default attention backend is SDPA. Set `attention` to `flash_attention_2` after installing a compatible FlashAttention build.

On each of eight nodes, run:

```bash
torchrun --nnodes=8 --nproc_per_node=8 \
  --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port=29500 \
  -m yuanbao_reranker.train \
  --config configs/train-0.6B.json \
  --data /path/to/train.jsonl \
  --output /shared/output/yuanbao-reranker-0.6B
```

For 4B or 8B, select the matching configuration. To train with SFT alone, pass `--rank-weight 0`; `teacher_scores` can then be omitted. To continue a saved training run, pass `--resume /path/to/checkpoint`.
