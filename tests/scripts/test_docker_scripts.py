"""Smoke tests for Docker workflow scripts."""

from __future__ import annotations

import subprocess
from pathlib import Path

DOCKER_SCRIPTS = [
    "bootstrap_repo.sh",
    "build.sh",
    "shell.sh",
    "check.sh",
    "fix.sh",
    "validate_dump.sh",
    "train_stage2.sh",
    "eval_stage2.sh",
    "export_teacher_dump.sh",
]

PORTAINER_SCRIPTS = [
    "export_teacher_dump_all.sh",
]


def test_dockerfile_uses_unified_cuda_image() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel" in dockerfile
    assert "HF_HOME=/cache/huggingface" in dockerfile
    assert "PYTHONPATH=/workspace/repo:/workspace/repo/alpamayo1.5/src" in dockerfile
    assert "REPO_URL=https://github.com/oakley-Thomas/CSE676-Project.git" in dockerfile
    assert 'ENTRYPOINT ["/usr/local/bin/bootstrap_repo.sh"]' in dockerfile
    assert "COPY . /workspace" not in dockerfile


def test_bootstrap_clones_repo_and_initializes_alpamayo_submodule() -> None:
    bootstrap = Path("scripts/docker/bootstrap_repo.sh").read_text(encoding="utf-8")
    assert "GIT_ASKPASS" in bootstrap
    assert "run_git clone --filter=blob:none" in bootstrap
    assert 'run_git -C "${repo_dir}" fetch --depth' in bootstrap
    assert 'run_git -C "${repo_dir}" submodule update' in bootstrap
    assert '-- "${ALPAMAYO_SUBMODULE_PATH}"' in bootstrap
    assert 'PYTHONPATH="${repo_dir}:${repo_dir}/${ALPAMAYO_SUBMODULE_PATH}/src"' in bootstrap


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


def test_portainer_scripts_are_bash_syntax_valid() -> None:
    for script_name in PORTAINER_SCRIPTS:
        script_path = Path("scripts/portainer") / script_name
        subprocess.run(["bash", "-n", str(script_path)], check=True)


def test_export_wrapper_runs_public_export_cli() -> None:
    wrapper = Path("scripts/docker/export_teacher_dump.sh").read_text(encoding="utf-8")
    assert "python -m scripts.export_teacher_dump" in wrapper
    assert "--num-traj-samples" in wrapper
    assert "--top-k" in wrapper


def test_portainer_stack_uses_external_persistent_volumes() -> None:
    stack = Path("deploy/portainer/teacher-export.stack.yml").read_text(encoding="utf-8")
    assert "alpamayo_hf_cache:/cache/huggingface" in stack
    assert "alpamayo_data:/workspace/data" in stack
    assert "external: true" in stack
    assert "HF_TOKEN" in stack
    assert "GITHUB_TOKEN" in stack
    assert "REPO_URL" in stack
    assert "REPO_REF" in stack
    assert 'command: ["scripts/portainer/export_teacher_dump_all.sh"]' in stack
    assert "gpus: all" in stack
