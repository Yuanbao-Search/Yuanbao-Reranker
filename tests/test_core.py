import json
import math
from pathlib import Path
import tempfile
import os
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

import torch

from yuanbao_reranker.aggregate import aggregate
from yuanbao_reranker.benchmark import inventory, prompts
from yuanbao_reranker.data import GroupCollator
from yuanbao_reranker.evaluate import evaluate, evaluation_settings
from yuanbao_reranker.losses import ranking_loss
from yuanbao_reranker.metrics import ndcg_at_10
from yuanbao_reranker.model import relevance_score, YuanbaoReranker
from yuanbao_reranker.prepare import prepare
from yuanbao_reranker.prompts import MLDR_INSTRUCTION, SUFFIX, encode_pair, encode_training_pair


class Tokenizer:
    pad_token_id = 0
    def encode(self, text, add_special_tokens=False):
        import re
        parts = re.split(r"(yes|no)", text)
        return [value for part in parts for value in
                ([3] if part == "yes" else [4] if part == "no" else [10 + ord(c) % 50 for c in part])]



def tiny_checkpoint(path):
    from tokenizers import Tokenizer as FastTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    words = ["[UNK]", "[PAD]", "[EOS]", "yes", "no", "short", "long", "document", "query"]
    tokenizer = FastTokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                                       pad_token="[PAD]", eos_token="[EOS]")
    tokenizer.save_pretrained(path)
    torch.manual_seed(7)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=len(words), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=512, pad_token_id=1, eos_token_id=2,
    ))
    model.save_pretrained(path)


class CoreTests(unittest.TestCase):
    def test_task_prompt_prepare_evaluate_and_score(self):
        from yuanbao_reranker.score import main
        tasks = ["TempReasonL1", "SpartQA"]
        instruction = "Given the user query, retrieval the relevant passages"
        for task in tasks:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as tmp:
                self.assertEqual(prompts()[task], instruction)
                unit = next(u for u in inventory() if u["task"] == task)
                root = Path(tmp)
                (root / "corpus.jsonl").write_text('{"id":"a","text":"document"}\n')
                (root / "queries.jsonl").write_text('{"id":"q","text":"query"}\n')
                (root / "qrels.json").write_text('{"q":{"a":1}}')
                (root / "candidates.json").write_text('{"q":{"a":1}}')
                prepare(unit, root / "corpus.jsonl", root / "queries.jsonl", root / "qrels.json",
                        root / "candidates.json", root / "prepared", False)
                model = Mock(max_length=8192, runtime={})
                model.score.return_value = [0.75]
                result = evaluate(model, root / "prepared", root / "result", "test")
                self.assertEqual(result["instruction"], instruction)
                self.assertEqual(model.score.call_args.args[1], instruction)
                model.score.reset_mock()
                with patch("yuanbao_reranker.score.YuanbaoReranker", return_value=model), redirect_stdout(StringIO()):
                    result = main(["--model", "weights", "--query", "query", "--doc", "document",
                                   "--task", task, "--benchmark", unit["benchmark"]])
                self.assertEqual(result["instruction"], instruction)
                self.assertEqual(model.score.call_args.args[1], instruction)


    def test_score(self):
        values = {3: SimpleNamespace(logprob=-1), 4: SimpleNamespace(logprob=-3)}
        self.assertAlmostEqual(relevance_score(values, 3, 4), 1 / (1 + math.exp(-2)))
        self.assertAlmostEqual(relevance_score({3: values[3]}, 3, 4), 1 / (1 + math.exp(-9)))
        self.assertEqual(relevance_score({}, 3, 4), 0.5)

    def test_suffix_survives_truncation(self):
        tokenizer = Tokenizer()
        ids = encode_pair(tokenizer, "q", "document" * 100, "instruction", 30)
        suffix = tokenizer.encode(SUFFIX)
        self.assertEqual(ids[-len(suffix):], suffix)
        self.assertEqual(len(ids), 30 + len(suffix))

    def test_training_masks_source(self):
        row = encode_training_pair(Tokenizer(), "q", "d" * 200, "instruction", True, 100)
        self.assertEqual(len(row["input_ids"]), 100)
        self.assertEqual(row["labels"][row["score_position"] + 1], 3)
        self.assertTrue(all(x == -100 for x in row["labels"][:20]))
        self.assertEqual(row["labels"][-1], Tokenizer().encode("\n")[0])

    def test_long_doc_prompt_encoding(self):
        from yuanbao_reranker.prompts import PREFIX
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
        text = PREFIX + f"<Instruct>: {instruction}\n\n<Query>: q\n\n<Document>: d"
        tokenizer = Tokenizer()
        self.assertEqual(encode_pair(tokenizer, "q", "d", prompts()["MultiLongDocRetrieval"]),
                         tokenizer.encode(text) + tokenizer.encode(SUFFIX))

    def test_groups_and_code_instruction(self):
        groups = [dict(query="implement f", instruction="Find code", positive="def f(): pass",
                       negatives=["a", "b", "c"], teacher_scores=[5, 2, 1, 0]) for _ in range(2)]
        batch = GroupCollator(Tokenizer(), 256)(groups)
        self.assertEqual(batch["input_ids"].shape[0], 8)
        self.assertEqual(batch["teacher_scores"].shape, (2, 4))
        rows = torch.arange(8)
        labels = batch["labels"][rows, batch["score_positions"] + 1]
        self.assertEqual(labels.tolist(), [3, 4, 4, 4, 3, 4, 4, 4])

    def test_missing_teacher_rejected(self):
        group = dict(query="q", instruction="i", positive="p", negatives=["a", "b", "c"])
        with self.assertRaises(ValueError):
            GroupCollator(Tokenizer(), 256)([group])
        batch = GroupCollator(Tokenizer(), 256, False)([group])
        self.assertEqual(batch["teacher_scores"].sum().item(), 0)

    def test_hinge_cap_and_pair_mean(self):
        student = torch.zeros(1, 4, requires_grad=True)
        teacher = torch.tensor([[10.0, 3, 1, 0]])
        loss = ranking_loss(student, teacher)
        self.assertAlmostEqual(loss.item(), (4 + 4 + 4 + 2 + 3 + 1) / 6)
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertLess(student.grad[0, 0].item(), 0)

    def test_satisfied_margins_zero(self):
        teacher = torch.tensor([[4.0, 3, 2, 0]])
        self.assertEqual(ranking_loss(teacher * 2, teacher).item(), 0)

    def test_teacher_conflict_and_ties(self):
        student = torch.zeros(3, 4, requires_grad=True)
        teacher = torch.tensor([[0.0, 1, 0, 0], [1, 1, 1, 1], [4, 3, 2, 1]])
        eligible = torch.tensor([True, True, False])
        loss = ranking_loss(student, teacher, eligible)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(student.grad.abs().sum().item(), 0)

    def test_invalid_groups_keep_denominator(self):
        teacher = torch.tensor([[4.0, 3, 2, 1], [0, 1, 0, 0]])
        student = torch.zeros_like(teacher)
        self.assertAlmostEqual(ranking_loss(student, teacher).item(), ranking_loss(student[:1], teacher[:1]).item() / 2)

    def test_linear_ndcg_and_unretrieved_positive(self):
        qrels = {"q": {"a": 3, "b": 1, "c": 2}}
        predictions = {"q": {"b": 1.0, "a": 0.5}}
        expected = (1 + 3 / math.log2(3)) / (3 + 2 / math.log2(3) + 1 / math.log2(4))
        self.assertEqual(ndcg_at_10(qrels, predictions), round(expected, 5))

    def test_missing_query_and_self_document(self):
        qrels = {"q": {"q": 1}, "r": {"x": 1}}
        self.assertEqual(ndcg_at_10(qrels, {"q": {"q": 1.0}}), 0.5)
        self.assertEqual(ndcg_at_10(qrels, {"q": {"q": 1.0}}, True), 0)

    def test_inventory_and_mldr_instruction(self):
        units = inventory()
        self.assertEqual(len(units), 135)
        expected = {"CMTEB-R": 8, "MMTEB-R": 49, "BEIR": 26, "MLDR": 13, "MTEB-Code": 39}
        self.assertEqual({panel: sum(x["benchmark"] == panel for x in units) for panel in expected}, expected)
        for unit in units:
            if unit["benchmark"] == "MLDR":
                self.assertEqual(prompts()[unit["task"]], MLDR_INSTRUCTION)

    def test_aggregation_hierarchy(self):
        results = [dict(unit, model="test", ndcg_at_10=0.0, instruction=prompts()[unit["task"]], max_length=8192, top_k=100) for unit in inventory()]
        for row in results:
            if row["benchmark"] == "BEIR" and row["task"].startswith("CQADupstack"):
                row["ndcg_at_10"] = 1.0
            if row["benchmark"] == "MMTEB-R" and row["task"] == "MIRACLRetrievalHardNegatives":
                row["ndcg_at_10"] = 1.0
        scores = aggregate(results)
        self.assertAlmostEqual(scores["panels"]["BEIR"], 100 / 15)
        self.assertAlmostEqual(scores["panels"]["MMTEB-R"], 100 / 16)
        with self.assertRaises(ValueError):
            aggregate(results[:-1])
        with self.assertRaises(ValueError):
            aggregate(results + results[:1])

    def test_prepare_evaluate_roundtrip(self):
        unit = inventory()[0]
        class Model:
            max_length = 8192
            def score(self, pairs, instruction, batch_size):
                return [0.9 if d == "relevant" else 0.1 for q, d in pairs]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "corpus.jsonl").write_text('{"id":"a","text":"relevant"}\n{"id":"b","text":"irrelevant"}\n')
            (root / "queries.jsonl").write_text('{"id":"q","text":"question"}\n')
            (root / "qrels.json").write_text('{"q":{"a":1}}')
            (root / "candidates.json").write_text('{"q":{"b":1,"a":0.5}}')
            prepare(unit, root / "corpus.jsonl", root / "queries.jsonl", root / "qrels.json", root / "candidates.json", root / "prepared", False)
            result = evaluate(Model(), root / "prepared", root / "result", "test")
            self.assertEqual(result["ndcg_at_10"], 1)

    def test_rerank_order(self):
        model = object.__new__(YuanbaoReranker)
        model.score = lambda pairs, instruction, batch_size: [0.2, 0.8, 0.8]
        self.assertEqual([r["index"] for r in model.rerank("q", ["a", "b", "c"], top_n=2)], [1, 2])

    def test_evaluation_batch_crosses_query_boundary(self):
        unit = inventory()[0]
        calls = []
        class Model:
            max_length = 8192
            def score(self, pairs, instruction, batch_size):
                calls.append(pairs)
                return [0.8 if document == "A" else 0.2 for query, document in pairs]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "corpus.jsonl").write_text('{"id":"a","text":"A"}\n{"id":"b","text":"B"}\n')
            (root / "queries.jsonl").write_text('{"id":"q1","text":"first"}\n{"id":"q2","text":"second"}\n')
            (root / "qrels.json").write_text('{"q1":{"a":1},"q2":{"a":1}}')
            (root / "candidates.json").write_text('{"q1":{"b":1,"a":0.5},"q2":{"a":1,"b":0.5}}')
            prepare(unit, root / "corpus.jsonl", root / "queries.jsonl", root / "qrels.json",
                    root / "candidates.json", root / "prepared", False)
            result = evaluate(Model(), root / "prepared", root / "result", "test", batch_size=3)
            self.assertEqual(calls, [[("first", "B"), ("first", "A"), ("second", "A")], [("second", "B")]])
            self.assertEqual(result["ndcg_at_10"], 1.0)

    def test_model_panel_settings(self):
        configs = Path(__file__).resolve().parents[1] / "configs"
        settings, environment = evaluation_settings(configs / "eval-0.6B.json", "MLDR")
        self.assertEqual((settings["tensor_parallel_size"], settings["batch_size"], settings["max_num_seqs"]), (8, 16, 512))
        settings, _ = evaluation_settings(configs / "eval-0.6B.json", "MTEB-Code")
        self.assertEqual((settings["tensor_parallel_size"], settings["batch_size"]), (1, 512))
        settings, _ = evaluation_settings(configs / "eval-4B.json", "BEIR", {"batch_size": 64, "dtype": None})
        self.assertEqual((settings["batch_size"], settings["dtype"]), (64, "float16"))
        self.assertEqual(environment["VLLM_USE_V1"], "1")


    def test_invalid_backend(self):
        with self.assertRaisesRegex(ValueError, "backend"):
            YuanbaoReranker("unused", backend="invalid")

    def test_vllm_defaults_and_score_unchanged(self):
        values = {3: SimpleNamespace(logprob=-1), 4: SimpleNamespace(logprob=-3)}
        output = SimpleNamespace(outputs=[SimpleNamespace(logprobs=[values])])
        llm = Mock()
        llm.return_value.generate.return_value = [output]
        sampling = Mock()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()), \
             patch.dict(sys.modules, {"vllm": SimpleNamespace(LLM=llm, SamplingParams=sampling),
                                      "vllm.inputs": SimpleNamespace(TokensPrompt=dict)}):
            model = YuanbaoReranker("local-model")
            llm.assert_called_once_with(model="local-model", trust_remote_code=False,
                dtype="float16", tensor_parallel_size=1, gpu_memory_utilization=0.8,
                max_model_len=10000, max_num_seqs=512, enable_prefix_caching=True,
                distributed_executor_backend="uni")
            sampling.assert_called_once_with(temperature=0, top_p=0.95, max_tokens=1,
                                             logprobs=20, allowed_token_ids=[3, 4])
            self.assertEqual(model.score([("q", "d")]), [relevance_score(values, 3, 4)])
            llm.return_value.generate.assert_called_once_with(
                [dict(prompt_token_ids=encode_pair(Tokenizer(), "q", "d", MLDR_INSTRUCTION))],
                sampling.return_value, use_tqdm=False)

    def test_transformers_padding_batch_and_probability(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"vllm": None}):
            tiny_checkpoint(tmp)
            model = YuanbaoReranker(tmp, backend="transformers", dtype="float32", device="cpu")
            pairs = [("query", "short"), ("query", "long document " * 5), ("query", "document")]
            expected = []
            with torch.inference_mode():
                for q, d in pairs:
                    ids = torch.tensor([encode_pair(model.tokenizer, q, d, MLDR_INSTRUCTION)])
                    logits = model.model(input_ids=ids).logits[0, -1].double()
                    scores = torch.log_softmax(logits, dim=-1)
                    expected.append(relevance_score({
                        model.yes_id: SimpleNamespace(logprob=scores[model.yes_id].item()),
                        model.no_id: SimpleNamespace(logprob=scores[model.no_id].item()),
                    }, model.yes_id, model.no_id))
            for side in ["left", "right"]:
                model.tokenizer.padding_side = side
                for batch_size in [1, 2, 3]:
                    scores = model.score(pairs, batch_size=batch_size)
                    for actual, target in zip(scores, expected):
                        self.assertAlmostEqual(actual, target, places=6)
                        self.assertTrue(0 <= actual <= 1)
            self.assertEqual(model.score([]), [])
            with self.assertRaises(ValueError):
                model.score(pairs, batch_size=0)
            order = model.rerank("query", [d for _, d in pairs])
            self.assertEqual([r["index"] for r in order], sorted(range(3), key=lambda i: (-expected[i], i)))
            example = Path(__file__).resolve().parents[1] / "examples/rerank.py"
            result = subprocess.run([sys.executable, str(example), "--model", tmp,
                                     "--backend", "transformers"], capture_output=True,
                                    text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = json.loads(result.stdout)
            self.assertEqual(len(output), 2)
            self.assertGreaterEqual(output[0]["relevance_score"], output[1]["relevance_score"])

    def test_toy_data_and_training_cli(self):
        root = Path(__file__).resolve().parents[1]
        data_path = root / "examples/toy_train.jsonl"
        groups = [json.loads(line) for line in data_path.read_text().splitlines()]
        self.assertTrue(4 <= len(groups) <= 8)
        for group in groups:
            self.assertEqual(set(group), {"query", "instruction", "positive", "negatives", "teacher_scores", "eligible"})
            self.assertEqual(len(group["negatives"]), 3)
            self.assertEqual(len(group["teacher_scores"]), 4)
            self.assertIsInstance(group["eligible"], bool)
        batch = GroupCollator(Tokenizer(), 256)(groups)
        self.assertEqual(batch["input_ids"].shape[0], len(groups) * 4)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tiny_checkpoint(tmp / "model")
            config = json.loads((root / "configs/train-toy.json").read_text())
            config["model"] = str(tmp / "model")
            config["training"].update(use_cpu=True, disable_tqdm=True)
            (tmp / "train.json").write_text(json.dumps(config))
            env = dict(os.environ, OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false",
                       HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
            result = subprocess.run([sys.executable, "-m", "yuanbao_reranker.train",
                "--config", str(tmp / "train.json"), "--data", str(data_path),
                "--output", str(tmp / "trained")], env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            state = json.loads((tmp / "trained/trainer_state.json").read_text())
            self.assertEqual(state["global_step"], 2)
            self.assertTrue(all(math.isfinite(row["loss"]) for row in state["log_history"] if "loss" in row))
            self.assertTrue((tmp / "trained/model.safetensors").is_file())
            self.assertTrue((tmp / "trained/tokenizer.json").is_file())



    def test_score_entry_uses_evaluation_config_and_prompts(self):
        from yuanbao_reranker.score import main
        root = Path(__file__).resolve().parents[1]
        for task, panel in [("MultiLongDocRetrieval", "MLDR"), ("ArguAna", "BEIR"),
                            ("ArguAna", "MMTEB-R"), ("T2Retrieval", "CMTEB-R"),
                            ("CodeEditSearchRetrieval", "MTEB-Code")]:
            config = root / "configs/eval-0.6B.json"
            settings, environment = evaluation_settings(config, panel)
            batch_size = settings.pop("batch_size")
            model = Mock(max_length=8192, runtime=settings)
            model.score.return_value = [0.75, 0.25]
            with patch("yuanbao_reranker.score.YuanbaoReranker", return_value=model) as constructor, redirect_stdout(StringIO()):
                result = main(["--model", "weights", "--query", "query", "--doc", "A", "--doc", "B",
                               "--task", task, "--benchmark", panel, "--config", str(config)])
            constructor.assert_called_once_with("weights", **settings)
            model.score.assert_called_once_with([("query", "A"), ("query", "B")], prompts()[task], batch_size)
            self.assertEqual(result["instruction"], prompts()[task])
            self.assertEqual(result["benchmark"], panel)
            self.assertEqual(result["backend"], "vllm")
            for key, value in environment.items():
                self.assertEqual(os.environ[key], value)

    def test_score_entry_rejects_mismatched_task_and_panel(self):
        from yuanbao_reranker.score import main
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            main(["--model", "unused", "--query", "q", "--doc", "d",
                  "--task", "MultiLongDocRetrieval", "--benchmark", "BEIR"])

    def test_score_entry_matches_evaluate_predictions(self):
        from yuanbao_reranker.score import main
        unit = next(u for u in inventory() if u["task"] == "MultiLongDocRetrieval")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tiny_checkpoint(root / "weights")
            model = YuanbaoReranker(root / "weights", backend="transformers", dtype="float32", device="cpu")
            (root / "corpus.jsonl").write_text('{"id":"a","text":"short"}\n{"id":"b","text":"long document"}\n')
            (root / "queries.jsonl").write_text('{"id":"q","text":"query"}\n')
            (root / "qrels.json").write_text('{"q":{"a":1}}')
            (root / "candidates.json").write_text('{"q":{"a":1,"b":0.5}}')
            prepare(unit, root / "corpus.jsonl", root / "queries.jsonl", root / "qrels.json",
                    root / "candidates.json", root / "prepared", False)
            evaluate(model, root / "prepared", root / "result", "tiny", batch_size=2)
            expected = json.loads((root / "result/predictions.json").read_text())["q"]
            with patch("yuanbao_reranker.score.YuanbaoReranker", return_value=model), redirect_stdout(StringIO()):
                result = main(["--model", "tiny", "--query", "query", "--doc", "short", "--doc", "long document",
                               "--task", "MultiLongDocRetrieval", "--backend", "transformers", "--batch-size", "2"])
            self.assertEqual([row["relevance_score"] for row in result["documents"]], [expected["a"], expected["b"]])
            process = subprocess.run([sys.executable, "-m", "yuanbao_reranker.score", "--model", str(root / "weights"),
                "--query", "query", "--doc", "short", "--doc", "long document", "--task", "MultiLongDocRetrieval",
                "--backend", "transformers", "--dtype", "float32", "--device", "cpu", "--batch-size", "2"],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            actual = json.loads(process.stdout)
            self.assertEqual([row["relevance_score"] for row in actual["documents"]], [expected["a"], expected["b"]])


if __name__ == "__main__":
    unittest.main()
