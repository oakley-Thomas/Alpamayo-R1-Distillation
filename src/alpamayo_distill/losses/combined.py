from __future__ import annotations

import torch

from alpamayo_distill.losses.kd_loss import kd_loss
from alpamayo_distill.losses.trajectory_loss import minade, trajectory_l2_loss


def combined_loss(outputs, batch: dict[str, torch.Tensor], config: dict) -> dict[str, torch.Tensor]:
    kd_cfg = config["kd"]
    loss_cfg = config["losses"]
    traj_cfg = loss_cfg["trajectory"]
    kd = kd_loss(
        student_logits=outputs.logits,
        teacher_topk_logits=batch["teacher_topk_logits"],
        teacher_topk_indices=batch["teacher_topk_indices"],
        temperature=float(kd_cfg["temperature"]),
        mask=batch.get("attention_mask"),
    )
    traj = trajectory_l2_loss(
        prediction=outputs.trajectory,
        target=batch["teacher_trajectory"],
        heading_weight=float(traj_cfg["heading_weight"]),
        horizon_decay=float(traj_cfg["horizon_decay"]),
        use_huber=bool(traj_cfg["use_huber"]),
        huber_delta=float(traj_cfg["huber_delta"]),
    )
    gt = minade(outputs.trajectory, batch["groundtruth_trajectory"])
    total = (
        float(loss_cfg["alpha_kd"]) * kd
        + float(loss_cfg["beta_trajectory"]) * traj
        + float(loss_cfg["gamma_groundtruth"]) * gt
    )
    return {"total": total, "kd": kd, "trajectory": traj, "groundtruth": gt}
