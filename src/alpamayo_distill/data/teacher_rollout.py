from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from alpamayo_distill.config import load_yaml
from alpamayo_distill.utils.logits_storage import LogitsStorage, quantize_topk_logits


@dataclass
class TeacherRolloutConfig:
    topk: int
    trajectory_steps: int
    cache_dir: Path
    cache_format: str
    seed: int
    vocab_size: int


class StubTeacher:
    def __init__(self, vocab_size: int, trajectory_steps: int, seed: int) -> None:
        self.vocab_size = vocab_size
        self.trajectory_steps = trajectory_steps
        self.rng = np.random.default_rng(seed)

    def infer(self, scene_id: str) -> dict[str, np.ndarray]:
        del scene_id
        seq_len = int(self.rng.integers(24, 96))
        full_logits = self.rng.normal(size=(seq_len, self.vocab_size)).astype(np.float32)
        token_ids = full_logits.argmax(axis=-1).astype(np.int64)
        trajectory = self.rng.normal(size=(self.trajectory_steps, 3)).astype(np.float32)
        return {"token_ids": token_ids, "full_logits": full_logits, "trajectory": trajectory}


def topk_from_logits(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argpartition(logits, -k, axis=-1)[..., -k:]
    topk_logits = np.take_along_axis(logits, idx, axis=-1)
    order = np.argsort(topk_logits, axis=-1)[..., ::-1]
    sorted_idx = np.take_along_axis(idx, order, axis=-1)
    sorted_logits = np.take_along_axis(topk_logits, order, axis=-1)
    return sorted_idx.astype(np.int32), sorted_logits.astype(np.float32)


def rollout_scene(scene_id: str, teacher: StubTeacher, storage: LogitsStorage, cfg: TeacherRolloutConfig) -> Path:
    teacher_out = teacher.infer(scene_id)
    topk_indices, topk_logits = topk_from_logits(teacher_out["full_logits"], cfg.topk)
    payload = {
        "scene_id": scene_id,
        "teacher_tokens": teacher_out["token_ids"],
        "teacher_topk_indices": topk_indices,
        "teacher_topk_logits": quantize_topk_logits(topk_logits),
        "logit_scale": storage.scale,
        "teacher_trajectory": teacher_out["trajectory"],
        "groundtruth_trajectory": teacher_out["trajectory"] + 0.05,
        "egomotion": np.zeros((4, 12), dtype=np.float32),
        "cameras": np.zeros((4, 4, 3, 320, 576), dtype=np.uint8),
        "scene_metadata": {"source_dataset": "stub", "scene_id": scene_id},
    }
    return storage.write_scene(scene_id, payload)


def load_rollout_config(data_cfg_path: str | Path, distill_cfg_path: str | Path) -> TeacherRolloutConfig:
    data_cfg = load_yaml(data_cfg_path)
    distill_cfg = load_yaml(distill_cfg_path)
    return TeacherRolloutConfig(
        topk=int(distill_cfg["kd"]["topk"]),
        trajectory_steps=int(distill_cfg["losses"]["trajectory"]["horizon_steps"]),
        cache_dir=Path(data_cfg["rollout"]["cache_dir"]),
        cache_format=str(data_cfg["rollout"]["cache_format"]),
        seed=int(data_cfg["rollout"]["seed"]),
        vocab_size=4096,
    )


def run_rollout(scene_ids: Iterable[str], cfg: TeacherRolloutConfig) -> list[Path]:
    teacher = StubTeacher(vocab_size=cfg.vocab_size, trajectory_steps=cfg.trajectory_steps, seed=cfg.seed)
    storage = LogitsStorage(cfg.cache_dir, fmt=cfg.cache_format)
    return [rollout_scene(scene_id, teacher, storage, cfg) for scene_id in scene_ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--distill-config", default="configs/distill.yaml")
    parser.add_argument("--num-scenes", type=int, default=8)
    args = parser.parse_args()
    cfg = load_rollout_config(args.data_config, args.distill_config)
    scene_ids = [f"scene-{index:06d}" for index in range(args.num_scenes)]
    written = run_rollout(scene_ids, cfg)
    print(f"Wrote {len(written)} teacher rollout scenes to {cfg.cache_dir}")


if __name__ == "__main__":
    main()
