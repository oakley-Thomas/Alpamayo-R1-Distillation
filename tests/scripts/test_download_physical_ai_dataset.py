"""Tests for the Physical AI dataset download helper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from huggingface_hub.errors import HfHubHTTPError

from scripts import download_physical_ai_dataset as downloader


@dataclass(frozen=True)
class _FakeResponse:
    status_code: int
    headers: dict[str, str]


class _FakeHfHubHTTPError(HfHubHTTPError):
    def __init__(self, status_code: int, headers: dict[str, str]) -> None:
        Exception.__init__(self, "fake Hub HTTP error")
        self.response = cast(Any, _FakeResponse(status_code=status_code, headers=headers))


def _config(tmp_path: Path) -> downloader.DownloadPhysicalAIConfig:
    return downloader.DownloadPhysicalAIConfig(
        repo_id="nvidia/PhysicalAI-Autonomous-Vehicles",
        revision=None,
        cache_dir=tmp_path / "hf-cache",
        allow_patterns=None,
        ignore_patterns=None,
        max_workers=1,
        rate_limit_retries=2,
        rate_limit_wait_seconds=99.0,
        require_token=True,
    )


def test_main_uses_rate_limit_safe_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_configs: list[downloader.DownloadPhysicalAIConfig] = []

    def fake_download(config: downloader.DownloadPhysicalAIConfig) -> Path:
        captured_configs.append(config)
        return tmp_path / "snapshot"

    monkeypatch.setattr(downloader, "download_physical_ai_dataset", fake_download)

    assert downloader.main([]) == 0
    assert captured_configs[0].max_workers == 1
    assert captured_configs[0].rate_limit_retries == 12
    assert captured_configs[0].rate_limit_wait_seconds == 310.0


def test_download_retries_and_reuses_cache_after_rate_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot_path = tmp_path / "snapshot"
    sleep_calls: list[float] = []
    call_count = 0

    def fake_snapshot_download(**kwargs: object) -> str:
        nonlocal call_count
        call_count += 1
        assert kwargs["max_workers"] == 1
        if call_count == 1:
            raise _FakeHfHubHTTPError(429, {"RateLimit": '"api";r=0;t=2'})
        return str(snapshot_path)

    monkeypatch.setattr(downloader, "get_token", lambda: "hf_test")
    monkeypatch.setattr(downloader, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(downloader.time, "sleep", sleep_calls.append)

    result = downloader.download_physical_ai_dataset(_config(tmp_path))

    assert result == snapshot_path
    assert call_count == 2
    assert sleep_calls == [7.0]


def test_download_does_not_retry_non_rate_limit_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    call_count = 0

    def fake_snapshot_download(**_: object) -> str:
        nonlocal call_count
        call_count += 1
        raise _FakeHfHubHTTPError(500, {})

    monkeypatch.setattr(downloader, "get_token", lambda: "hf_test")
    monkeypatch.setattr(downloader, "snapshot_download", fake_snapshot_download)

    with pytest.raises(HfHubHTTPError):
        downloader.download_physical_ai_dataset(_config(tmp_path))

    assert call_count == 1
