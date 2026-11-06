import math
from types import SimpleNamespace

from .prompts import DEFAULT_INSTRUCTION, encode_pair, token_id


def relevance_score(logprobs, yes_id, no_id):
    def value(token):
        result = logprobs.get(token)
        return -10.0 if result is None else float(result.logprob)
    yes, no = value(yes_id), value(no_id)
    if not math.isfinite(yes) or not math.isfinite(no):
        raise ValueError("Non-finite token score")
    yes, no = math.exp(yes), math.exp(no)
    if yes + no == 0:
        raise ValueError("Both token probabilities underflowed")
    return yes / (yes + no)


class YuanbaoReranker:
    def __init__(self, model, max_length=8192, tensor_parallel_size=1, gpu_memory_utilization=0.8,
                 dtype="float16", max_model_len=10000, max_num_seqs=512,
                 enable_prefix_caching=True, distributed_executor_backend="uni",
                 backend="vllm", device=None):
        if backend not in {"vllm", "transformers"}:
            raise ValueError("backend must be 'vllm' or 'transformers'")
        from transformers import AutoTokenizer
        self.backend = backend
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
        self.yes_id = token_id(self.tokenizer, "yes")
        self.no_id = token_id(self.tokenizer, "no")
        self.max_length = max_length
        if backend == "transformers":
            import torch
            from transformers import AutoModelForCausalLM
            dtypes = {"float16": torch.float16, "half": torch.float16,
                      "bfloat16": torch.bfloat16, "float32": torch.float32,
                      "float": torch.float32, "auto": "auto"}
            if dtype not in dtypes:
                raise ValueError(f"Unsupported dtype: {dtype}")
            if self.tokenizer.pad_token_id is None:
                if self.tokenizer.eos_token_id is None:
                    raise ValueError("Tokenizer must define a padding or EOS token")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = AutoModelForCausalLM.from_pretrained(
                model, trust_remote_code=False, torch_dtype=dtypes[dtype],
            ).to(device).eval()
            self.runtime = dict(backend=backend, dtype=dtype, device=str(self.model.device))
            return
        from vllm import LLM, SamplingParams
        self.runtime = dict(dtype=dtype, tensor_parallel_size=tensor_parallel_size,
                            gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
                            max_num_seqs=max_num_seqs, enable_prefix_caching=enable_prefix_caching,
                            distributed_executor_backend=distributed_executor_backend)
        self.model = LLM(
            model=model, trust_remote_code=False, **self.runtime,
        )
        self.sampling = SamplingParams(
            temperature=0, top_p=0.95, max_tokens=1, logprobs=20,
            allowed_token_ids=[self.yes_id, self.no_id],
        )

    def score(self, pairs, instruction=DEFAULT_INSTRUCTION, batch_size=32):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.backend == "transformers":
            return self._score_transformers(pairs, instruction, batch_size)
        from vllm.inputs import TokensPrompt
        result = []
        for start in range(0, len(pairs), batch_size):
            prompts = [TokensPrompt(prompt_token_ids=encode_pair(
                self.tokenizer, q, d, instruction, self.max_length
            )) for q, d in pairs[start:start + batch_size]]
            outputs = self.model.generate(prompts, self.sampling, use_tqdm=False)
            if len(outputs) != len(prompts):
                raise RuntimeError("Incomplete model output")
            for output in outputs:
                result.append(relevance_score(output.outputs[0].logprobs[-1], self.yes_id, self.no_id))
        return result

    def _score_transformers(self, pairs, instruction, batch_size):
        import torch
        result = []
        with torch.inference_mode():
            for start in range(0, len(pairs), batch_size):
                encoded = [encode_pair(self.tokenizer, q, d, instruction, self.max_length)
                           for q, d in pairs[start:start + batch_size]]
                batch = self.tokenizer.pad({"input_ids": encoded}, padding=True,
                                           return_attention_mask=True, return_tensors="pt")
                batch = {key: value.to(self.model.device) for key, value in batch.items()}
                mask = batch["attention_mask"]
                positions = mask.long().cumsum(-1) - 1
                positions.masked_fill_(mask == 0, 0)
                logits = self.model(**batch, position_ids=positions, use_cache=False).logits
                last = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
                last = last.masked_fill(mask == 0, -1).max(dim=1).values
                rows = torch.arange(len(encoded), device=mask.device)
                selected = logits[rows, last][:, [self.yes_id, self.no_id]].float()
                if not torch.isfinite(selected).all():
                    raise ValueError("Non-finite token score")
                logprobs = selected.log_softmax(dim=-1).cpu().tolist()
                for yes, no in logprobs:
                    result.append(relevance_score({
                        self.yes_id: SimpleNamespace(logprob=yes),
                        self.no_id: SimpleNamespace(logprob=no),
                    }, self.yes_id, self.no_id))
        return result

    def rerank(self, query, documents, instruction=DEFAULT_INSTRUCTION, top_n=None, batch_size=32):
        if top_n is not None and top_n < 0:
            raise ValueError("top_n must be nonnegative")
        scores = self.score([(query, d) for d in documents], instruction, batch_size)
        order = sorted(range(len(documents)), key=lambda i: (-scores[i], i))
        if top_n is not None:
            order = order[:top_n]
        return [{"index": i, "document": documents[i], "relevance_score": scores[i]} for i in order]
