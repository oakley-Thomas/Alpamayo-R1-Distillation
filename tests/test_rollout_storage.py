from __future__ import annotations

from pathlib import Path
import unittest

from alpamayo_distill.data.teacher_rollout import TeacherRolloutConfig, run_rollout
from alpamayo_distill.utils.logits_storage import LogitsStorage


class RolloutStorageTest(unittest.TestCase):
    def test_stub_rollout_roundtrip(self) -> None:
        tmp_path = Path("tmp_test_rollout")
        tmp_path.mkdir(exist_ok=True)
        cfg = TeacherRolloutConfig(
            topk=8,
            trajectory_steps=12,
            cache_dir=tmp_path,
            cache_format="mmap_npy",
            seed=7,
            vocab_size=128,
        )
        written = run_rollout(["scene-000001"], cfg)
        self.assertEqual(len(written), 1)
        storage = LogitsStorage(tmp_path, fmt="mmap_npy")
        payload = storage.read_scene(written[0])
        self.assertEqual(payload["teacher_topk_indices"].shape, (payload["teacher_tokens"].shape[0], 8))
        self.assertEqual(payload["teacher_trajectory"].shape, (12, 3))
        written[0].unlink(missing_ok=True)
        tmp_path.rmdir()


if __name__ == "__main__":
    unittest.main()
