"""Download the NVIDIA Physical AI AV dataset into a Hugging Face cache."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import get_token, snapshot_download

DEFAULT_DATASET_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"


@dataclass(frozen=True)
class DownloadPhysicalAIConfig:
    """Configuration for downloading the gated Physical AI AV dataset."""

    repo_id: str
    revision: str | None
    cache_dir: Path | None
    allow_patterns: tuple[str, ...] | None
    ignore_patterns: tuple[str, ...] | None
    max_workers: int
    require_token: bool


def _patterns(values: list[str] | None) -> tuple[str, ...] | None:
    """Convert argparse pattern lists into the shape expected by Hugging Face Hub."""
    if values is None:
        return None
    return tuple(values)


def _parse_args(argv: list[str] | None) -> DownloadPhysicalAIConfig:
    """Parse command-line arguments for the dataset downloader."""
    parser = argparse.ArgumentParser(
        description=(
            "Download NVIDIA's gated PhysicalAI-Autonomous-Vehicles dataset into "
            "the active Hugging Face cache. In Docker, mount the target volume at "
            "HF_HOME so teacher inference can reuse the cached files."
        )
    )
    parser.add_argument("--repo-id", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional Hugging Face Hub cache directory. When omitted, "
            "huggingface_hub uses HF_HOME/HF_HUB_CACHE."
        ),
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        dest="allow_patterns",
        help="Optional file pattern to include. May be passed multiple times.",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        dest="ignore_patterns",
        help="Optional file pattern to exclude. May be passed multiple times.",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--no-require-token",
        action="store_true",
        help="Skip the early token check. The Hub request may still fail for gated data.",
    )
    args = parser.parse_args(argv)
    return DownloadPhysicalAIConfig(
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        allow_patterns=_patterns(args.allow_patterns),
        ignore_patterns=_patterns(args.ignore_patterns),
        max_workers=args.max_workers,
        require_token=not args.no_require_token,
    )


def download_physical_ai_dataset(config: DownloadPhysicalAIConfig) -> Path:
    """Download the dataset snapshot into the configured Hugging Face cache.

    Args:
        config: Download configuration. The default target is the Hub cache
            under ``HF_HOME``, which is what ``physical_ai_av`` and teacher
            inference use when loading clips.

    Returns:
        Path to the cached dataset snapshot.
    """
    if config.require_token and get_token() is None:
        raise RuntimeError(
            "No Hugging Face token found. Set HF_TOKEN or run `hf auth login` "
            "inside a container that has the target HF_HOME volume mounted."
        )
    token: bool | None = True if config.require_token else None

    snapshot_path = snapshot_download(
        repo_id=config.repo_id,
        repo_type="dataset",
        revision=config.revision,
        cache_dir=None if config.cache_dir is None else str(config.cache_dir),
        allow_patterns=config.allow_patterns,
        ignore_patterns=config.ignore_patterns,
        max_workers=config.max_workers,
        token=token,
    )
    return Path(snapshot_path)


def main(argv: list[str] | None = None) -> int:
    """Run the Physical AI dataset downloader CLI."""
    config = _parse_args(argv)
    try:
        snapshot_path = download_physical_ai_dataset(config)
    except Exception as exc:
        print(f"Dataset download failed: {exc}", file=sys.stderr)
        return 1

    hf_home = os.environ.get("HF_HOME", "<unset>")
    hf_hub_cache = os.environ.get("HF_HUB_CACHE", "<default>")
    print(f"Downloaded {config.repo_id}")
    print(f"HF_HOME={hf_home}")
    print(f"HF_HUB_CACHE={hf_hub_cache}")
    print(f"snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
