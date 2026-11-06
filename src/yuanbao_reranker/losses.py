import math

import torch


def ranking_loss(student, teacher, eligible=None, tie_epsilon=0.2, max_gap=4.0):
    if student.ndim != 2 or student.shape[0] < 1 or student.shape[1] != 4 or teacher.shape != student.shape:
        raise ValueError("Scores must have shape [groups, 4]")
    if not math.isfinite(tie_epsilon) or not math.isfinite(max_gap) or tie_epsilon < 0 or max_gap <= 0:
        raise ValueError("Invalid ranking loss parameters")
    student = student.float()
    teacher = teacher.detach().to(device=student.device, dtype=torch.float32)
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError("Scores must be finite")
    valid = (teacher[:, 1:] <= teacher[:, :1]).all(dim=1)
    if eligible is not None:
        if eligible.shape != valid.shape:
            raise ValueError("Eligibility must have shape [groups]")
        valid = valid & eligible.to(device=student.device, dtype=torch.bool)
    left, right = torch.triu_indices(4, 4, offset=1, device=student.device)
    gap = teacher[:, left] - teacher[:, right]
    mask = valid[:, None] & (gap.abs() > tie_epsilon)
    margin = gap.abs().clamp(max=max_gap)
    prediction = gap.sign() * (student[:, left] - student[:, right])
    losses = torch.relu(margin - prediction) * mask
    return (losses.sum(dim=1) / mask.sum(dim=1).clamp(min=1)).mean()
