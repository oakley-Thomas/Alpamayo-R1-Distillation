"""Smoke tests for Docker workflow scripts."""

from __future__ import annotations

import subprocess
from pathlib import Path

DOCKER_SCRIPTS = [
    "build.sh",
    "shell.sh",
    "check.sh",
    "fix.sh",
    "validate_dump.sh",
    "train_stage2.sh",
    "eval_stage2.sh",
    "export_teacher_dump.sh",
]


def test_dockerfile_uses_unified_cuda_image() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel" in dockerfile
    assert "HF_HOME=/cache/huggingface" in dockerfile
    assert "PYTHONPATH=/workspace:/workspace/alpamayo1.5/src" in dockerfile


def test_docker_shell_mounts_repo_and_hf_cache() -> None:
    shell_script = Path("scripts/docker/shell.sh").read_text(encoding="utf-8")
    assert "GPU_MODE=" in shell_script
    assert "--gpus all" in shell_script
    assert '-v "${REPO_ROOT}:/workspace"' in shell_script
    assert '-v "${HF_CACHE}:/cache/huggingface"' in shell_script


def test_cpu_only_wrappers_disable_gpu_mode() -> None:
    check_wrapper = Path("scripts/docker/check.sh").read_text(encoding="utf-8")
    fix_wrapper = Path("scripts/docker/fix.sh").read_text(encoding="utf-8")
    validate_wrapper = Path("scripts/docker/validate_dump.sh").read_text(encoding="utf-8")

    assert "GPU_MODE=none" in check_wrapper
    assert "GPU_MODE=none" in fix_wrapper
    assert "GPU_MODE=none" in validate_wrapper


def test_docker_scripts_are_bash_syntax_valid() -> None:
    for script_name in DOCKER_SCRIPTS:
        script_path = Path("scripts/docker") / script_name
        subprocess.run(["bash", "-n", str(script_path)], check=True)


def test_export_wrapper_runs_public_export_cli() -> None:
    wrapper = Path("scripts/docker/export_teacher_dump.sh").read_text(encoding="utf-8")
    assert "python -m scripts.export_teacher_dump" in wrapper
    assert "--num-traj-samples" in wrapper
    assert "--top-k" in wrapper
