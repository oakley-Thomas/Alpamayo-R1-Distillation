"""Validate teacher dump clips and split files."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.teacher_dump import ClipValidationError, TeacherDumpDataset


def _split_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.json"))


def main(argv: list[str] | None = None) -> int:
    """Validate all clips referenced by one split file or a split directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--include-kv-cache", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root)
    split_paths = _split_files(Path(args.splits))
    if not split_paths:
        print(f"No split JSON files found under {args.splits}")
        return 1

    try:
        for split_path in split_paths:
            dataset = TeacherDumpDataset(
                root=root,
                split_file=split_path,
                include_kv_cache=args.include_kv_cache,
            )
            print(f"{split_path}: validated {len(dataset)} clips")
    except ClipValidationError as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
