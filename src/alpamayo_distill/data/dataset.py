from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from alpamayo_distill.utils.logits_storage import LogitsStorage


class TeacherRolloutDataset(Dataset[dict[str, Any]]):
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.files = sorted(self.cache_dir.glob("*.h5")) + sorted(self.cache_dir.glob("*.npz"))
        self.storage = LogitsStorage(self.cache_dir)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        payload = self.storage.read_scene(self.files[index])
        return {
            "scene_id": payload["scene_id"],
            "cameras": torch.tensor(np.asarray(payload["cameras"]), dtype=torch.float32) / 255.0,
            "egomotion": torch.tensor(np.asarray(payload["egomotion"]), dtype=torch.float32),
            "teacher_tokens": torch.tensor(np.asarray(payload["teacher_tokens"]), dtype=torch.long),
            "teacher_topk_indices": torch.tensor(np.asarray(payload["teacher_topk_indices"]), dtype=torch.long),
            "teacher_topk_logits": torch.tensor(np.asarray(payload["teacher_topk_logits"]), dtype=torch.int16),
            "teacher_trajectory": torch.tensor(np.asarray(payload["teacher_trajectory"]), dtype=torch.float32),
            "groundtruth_trajectory": torch.tensor(np.asarray(payload["groundtruth_trajectory"]), dtype=torch.float32),
        }
