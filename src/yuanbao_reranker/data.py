import torch

from .prompts import encode_training_pair


class GroupCollator:
    def __init__(self, tokenizer, max_length=4096, require_teacher=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.require_teacher = require_teacher

    def __call__(self, groups):
        rows, teachers, eligible = [], [], []
        for group in groups:
            if len(group["negatives"]) != 3:
                raise ValueError("Each group must contain three negatives")
            instruction = group["instruction"]
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("Each group must provide its task instruction")
            for slot, document in enumerate([group["positive"], *group["negatives"]]):
                if not isinstance(document, str) or not isinstance(group["query"], str):
                    raise ValueError("Queries and documents must be strings")
                rows.append(encode_training_pair(
                    self.tokenizer, group["query"], document, instruction, slot == 0, self.max_length
                ))
            teacher = group.get("teacher_scores")
            if teacher is None:
                if self.require_teacher:
                    raise ValueError("teacher_scores are required for distillation")
                teacher = [0.0] * 4
            if len(teacher) != 4:
                raise ValueError("teacher_scores must follow positive, negative1, negative2, negative3")
            teachers.append(teacher)
            eligible.append(group.get("eligible", True))
        width = max(len(row["input_ids"]) for row in rows)
        pad = self.tokenizer.pad_token_id
        if pad is None:
            raise ValueError("Tokenizer must define a padding token")
        return {
            "input_ids": torch.tensor([r["input_ids"] + [pad] * (width - len(r["input_ids"])) for r in rows]),
            "attention_mask": torch.tensor([[1] * len(r["input_ids"]) + [0] * (width - len(r["input_ids"])) for r in rows]),
            "labels": torch.tensor([r["labels"] + [-100] * (width - len(r["labels"])) for r in rows]),
            "score_positions": torch.tensor([r["score_position"] for r in rows]),
            "teacher_scores": torch.tensor(teachers, dtype=torch.float32),
            "eligible": torch.tensor(eligible, dtype=torch.bool),
        }
