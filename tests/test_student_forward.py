from __future__ import annotations

import unittest

import torch

from alpamayo_distill.config import load_yaml, merge_dicts
from alpamayo_distill.losses.combined import combined_loss
from alpamayo_distill.models.student_vla import StudentVLA


class StudentForwardTest(unittest.TestCase):
    def test_student_forward_backward(self) -> None:
        cfg = load_yaml("configs/student.yaml")
        distill_cfg = load_yaml("configs/distill.yaml")
        cfg["processor"]["provider"] = "stub"
        cfg["backbone"]["provider"] = "stub"
        model = StudentVLA(cfg)
        batch = {
            "cameras": torch.randn(2, 4, 4, 3, 320, 576),
            "egomotion": torch.randn(2, 4, 12),
            "text": torch.randint(0, cfg["backbone"]["vocab_size"], (2, 32)),
            "teacher_topk_indices": torch.randint(0, cfg["backbone"]["vocab_size"], (2, 32, distill_cfg["kd"]["topk"])),
            "teacher_topk_logits": torch.randint(-1000, 1000, (2, 32, distill_cfg["kd"]["topk"]), dtype=torch.int16),
            "teacher_trajectory": torch.randn(2, 12, 3),
            "groundtruth_trajectory": torch.randn(2, 12, 3),
        }
        outputs = model(batch)
        self.assertEqual(outputs.logits.shape, (2, 32, cfg["backbone"]["vocab_size"]))
        self.assertEqual(outputs.trajectory.shape, (2, 12, 3))
        merged = merge_dicts(cfg, distill_cfg)
        losses = combined_loss(outputs, batch, merged)
        losses["total"].backward()
        grad_norm = sum(param.grad.norm().item() for param in model.parameters() if param.grad is not None)
        self.assertGreater(grad_norm, 0.0)


if __name__ == "__main__":
    unittest.main()
