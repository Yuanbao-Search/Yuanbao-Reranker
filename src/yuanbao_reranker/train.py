import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from .data import GroupCollator
from .losses import ranking_loss
from .prompts import token_id


class RankingTrainer(Trainer):
    def __init__(self, *args, rank_weight=0.1, tie_epsilon=0.2, max_gap=4.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank_weight = rank_weight
        self.tie_epsilon = tie_epsilon
        self.max_gap = max_gap
        self.yes_id = token_id(self.processing_class, "yes")
        self.no_id = token_id(self.processing_class, "no")
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        values = dict(inputs)
        positions = values.pop("score_positions")
        teacher = values.pop("teacher_scores")
        eligible = values.pop("eligible")
        outputs = model(**values, use_cache=False)
        loss = outputs.loss
        if self.rank_weight:
            rows = torch.arange(len(positions), device=outputs.logits.device)
            logits = outputs.logits[rows, positions].float()
            student = (logits[:, self.yes_id] - logits[:, self.no_id]).reshape(-1, 4)
            loss = loss + self.rank_weight * ranking_loss(
                student, teacher, eligible, self.tie_epsilon, self.max_gap
            )
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--rank-weight", type=float)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    set_seed(config["training"]["seed"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=False)
    settings = dict(config["training"])
    if settings.get("deepspeed"):
        settings["deepspeed"] = str(config_path.parent / settings["deepspeed"])
    settings["output_dir"] = args.output
    training_args = TrainingArguments(**settings)
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], torch_dtype=torch.bfloat16 if settings.get("bf16") else torch.float32,
        attn_implementation=config.get("attention", "sdpa"), trust_remote_code=False,
    )
    model.config.use_cache = False
    from datasets import load_dataset
    dataset = load_dataset("json", data_files=args.data, split="train")
    weight = config["rank_weight"] if args.rank_weight is None else args.rank_weight
    if weight < 0:
        raise ValueError("rank_weight must be nonnegative")
    trainer = RankingTrainer(
        model=model, args=training_args, processing_class=tokenizer, train_dataset=dataset,
        data_collator=GroupCollator(tokenizer, config["max_length"], weight > 0),
        rank_weight=weight, tie_epsilon=config["tie_epsilon"], max_gap=config["max_gap"],
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.output)
    trainer.save_state()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
