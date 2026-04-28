from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def dump_trajectory_json(path: str | Path, prediction: np.ndarray, target: np.ndarray) -> None:
    payload = {"prediction": prediction.tolist(), "target": target.tolist()}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
