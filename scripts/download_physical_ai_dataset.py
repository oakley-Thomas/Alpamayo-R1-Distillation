"""Download the NVIDIA Physical AI AV dataset into a Hugging Face cache."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# The Physical AI AV dataset has many small Xet-backed files. hf-xet asks the
# Hub API for short-lived read tokens and can exhaust the free API bucket before
# the cache is populated, so default to resolver downloads unless the caller
# explicitly sets HF_HUB_DISABLE_XET=0 before this module imports huggingface_hub.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import get_token, snapshot_download
from huggingface_hub.errors import HfHubHTTPError

DEFAULT_DATASET_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
DEFAULT_MAX_WORKERS = 1
DEFAULT_RATE_LIMIT_RETRIES = 12
DEFAULT_RATE_LIMIT_WAIT_SECONDS = 310.0
RATE_LIMIT_SAFETY_SECONDS = 5.0


@dataclass(frozen=True)
class DownloadPhysicalAIConfig:
    """Configuration for downloading the gated Physical AI AV dataset."""

    repo_id: str
    revision: str | None
    cache_dir: Path | None
    allow_patterns: tuple[str, ...] | None
    ignore_patterns: tuple[str, ...] | None
    max_workers: int
    rate_limit_retries: int
    rate_limit_wait_seconds: float
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
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=(
            "Number of concurrent Hub file downloads. Keep this low for the "
            "Physical AI dataset because it contains many small files."
        ),
    )
    parser.add_argument(
        "--rate-limit-retries",
        type=int,
        default=DEFAULT_RATE_LIMIT_RETRIES,
        help="Number of times to resume after a Hugging Face 429 response.",
    )
    parser.add_argument(
        "--rate-limit-wait-seconds",
        type=float,
        default=DEFAULT_RATE_LIMIT_WAIT_SECONDS,
        help=(
            "Fallback sleep duration after a 429 when the Hub response does "
            "not include a reset time."
        ),
    )
    parser.add_argument(
        "--no-require-token",
        action="store_true",
        help="Skip the early token check. The Hub request may still fail for gated data.",
    )
    args = parser.parse_args(argv)
    if args.max_workers < 1:
        parser.error("--max-workers must be >= 1")
    if args.rate_limit_retries < 0:
        parser.error("--rate-limit-retries must be >= 0")
    if args.rate_limit_wait_seconds <= 0.0:
        parser.error("--rate-limit-wait-seconds must be > 0")
    return DownloadPhysicalAIConfig(
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        allow_patterns=_patterns(args.allow_patterns),
        ignore_patterns=_patterns(args.ignore_patterns),
        max_workers=args.max_workers,
        rate_limit_retries=args.rate_limit_retries,
        rate_limit_wait_seconds=args.rate_limit_wait_seconds,
        require_token=not args.no_require_token,
    )


def _is_rate_limit_error(exc: HfHubHTTPError) -> bool:
    """Return whether a Hugging Face HTTP error is a 429 response."""
    status_code = getattr(exc.response, "status_code", None)
    return status_code == 429


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a numeric Retry-After header into seconds."""
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    if seconds <= 0.0:
        return None
    return seconds


def _parse_rate_limit_reset(value: str | None) -> float | None:
    """Parse the Hub RateLimit header reset timer into seconds."""
    if value is None:
        return None
    match = re.search(r"\bt=(\d+(?:\.\d+)?)", value)
    if match is None:
        return None
    seconds = float(match.group(1))
    if seconds <= 0.0:
        return None
    return seconds + RATE_LIMIT_SAFETY_SECONDS


def _rate_limit_wait_seconds(exc: HfHubHTTPError, fallback_seconds: float) -> float:
    """Choose a sleep duration from Hugging Face rate-limit headers."""
    headers = exc.response.headers
    return (
        _parse_retry_after(headers.get("Retry-After"))
        or _parse_rate_limit_reset(headers.get("RateLimit"))
        or fallback_seconds
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

    for attempt_idx in range(config.rate_limit_retries + 1):
        try:
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
        except HfHubHTTPError as exc:
            if not _is_rate_limit_error(exc) or attempt_idx == config.rate_limit_retries:
                raise
            wait_seconds = _rate_limit_wait_seconds(exc, config.rate_limit_wait_seconds)
            next_attempt = attempt_idx + 1
            print(
                "Hugging Face rate limit hit; "
                f"waiting {wait_seconds:.0f}s before retry "
                f"{next_attempt}/{config.rate_limit_retries}. "
                "Completed files remain in the local cache.",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)

    raise RuntimeError("unreachable dataset download retry state")


def _xet_mode() -> str:
    """Return the effective Xet toggle for logging."""
    return os.environ.get("HF_HUB_DISABLE_XET", "<unset>")


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
    print(f"HF_HUB_DISABLE_XET={_xet_mode()}")
    print(f"max_workers={config.max_workers}")
    print(f"snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
