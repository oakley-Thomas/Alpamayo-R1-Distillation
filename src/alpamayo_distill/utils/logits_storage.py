from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


def quantize_topk_logits(logits: np.ndarray, scale: float = 1000.0) -> np.ndarray:
    clipped = np.clip(logits * scale, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
    return clipped.astype(np.int16)


def dequantize_topk_logits(logits_q: np.ndarray, scale: float = 1000.0) -> np.ndarray:
    return logits_q.astype(np.float32) / scale


def scene_checksum(payload: dict[str, Any]) -> int:
    serializable = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return zlib.crc32(serializable) & 0xFFFFFFFF


class LogitsStorage:
    def __init__(self, root: str | Path, fmt: str = "hdf5", scale: float = 1000.0) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.fmt = fmt
        self.scale = scale

    def write_scene(self, scene_id: str, payload: dict[str, Any]) -> Path:
        payload = dict(payload)
        payload["checksum"] = scene_checksum(payload)
        if self.fmt == "hdf5":
            return self._write_hdf5(scene_id, payload)
        if self.fmt == "mmap_npy":
            return self._write_npz(scene_id, payload)
        raise ValueError(f"Unsupported cache format: {self.fmt}")

    def _write_hdf5(self, scene_id: str, payload: dict[str, Any]) -> Path:
        if h5py is None:
            raise RuntimeError("h5py is required for hdf5 cache format")
        path = self.root / f"{scene_id}.h5"
        with h5py.File(path, "w") as handle:
            for key, value in payload.items():
                if isinstance(value, np.ndarray):
                    handle.create_dataset(key, data=value)
                else:
                    handle.attrs[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
        return path

    def _write_npz(self, scene_id: str, payload: dict[str, Any]) -> Path:
        path = self.root / f"{scene_id}.npz"
        arrays: dict[str, np.ndarray] = {}
        metadata: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, np.ndarray):
                arrays[key] = value
            else:
                metadata[key] = value
        arrays["_metadata_json"] = np.array(json.dumps(metadata))
        np.savez_compressed(path, **arrays)
        return path

    def read_scene(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        if path.suffix == ".h5":
            return self._read_hdf5(path)
        if path.suffix == ".npz":
            return self._read_npz(path)
        raise ValueError(f"Unsupported file: {path}")

    def _read_hdf5(self, path: Path) -> dict[str, Any]:
        if h5py is None:
            raise RuntimeError("h5py is required for hdf5 cache format")
        payload: dict[str, Any] = {}
        with h5py.File(path, "r") as handle:
            for key in handle.keys():
                payload[key] = handle[key][()]
            for key, value in handle.attrs.items():
                if isinstance(value, str):
                    try:
                        payload[key] = json.loads(value)
                    except json.JSONDecodeError:
                        payload[key] = value
                else:
                    payload[key] = value
        return payload

    def _read_npz(self, path: Path) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        with np.load(path, allow_pickle=False) as archive:
            payload.update(json.loads(str(archive["_metadata_json"])))
            for key in archive.files:
                if key != "_metadata_json":
                    payload[key] = archive[key]
        return payload
