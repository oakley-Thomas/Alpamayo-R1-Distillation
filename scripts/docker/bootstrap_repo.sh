#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
MOUNTED_REPO="${MOUNTED_REPO:-${WORKSPACE_ROOT}}"
REPO_DIR="${REPO_DIR:-${WORKSPACE_ROOT}/repo}"
REPO_URL="${REPO_URL:-https://github.com/oakley-Thomas/CSE676-Project.git}"
REPO_REF="${REPO_REF:-VLM-Backbone}"
REPO_DEPTH="${REPO_DEPTH:-1}"
ALPAMAYO_SUBMODULE_PATH="${ALPAMAYO_SUBMODULE_PATH:-alpamayo1.5}"
INSTALL_REPO_EDITABLE="${INSTALL_REPO_EDITABLE:-true}"

repo_dir="${REPO_DIR}"
askpass_script=""

cleanup() {
    if [[ -n "${askpass_script}" && -f "${askpass_script}" ]]; then
        rm -f "${askpass_script}"
    fi
}
trap cleanup EXIT

if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    askpass_script="$(mktemp)"
    cat > "${askpass_script}" <<'EOF'
#!/usr/bin/env bash
case "$1" in
    *Username*) printf '%s\n' "x-access-token" ;;
    *Password*) printf '%s\n' "${GITHUB_TOKEN}" ;;
    *) printf '\n' ;;
esac
EOF
    chmod 700 "${askpass_script}"
fi

run_git() {
    if [[ -n "${askpass_script}" ]]; then
        GIT_ASKPASS="${askpass_script}" GIT_TERMINAL_PROMPT=0 git "$@"
    else
        git "$@"
    fi
}

if [[ -d "${MOUNTED_REPO}/.git" ]]; then
    repo_dir="${MOUNTED_REPO}"
else
    if [[ ! -d "${repo_dir}/.git" ]]; then
        mkdir -p "$(dirname "${repo_dir}")"
        run_git clone --filter=blob:none "${REPO_URL}" "${repo_dir}"
    fi

    if [[ -n "${REPO_REF}" ]]; then
        run_git -C "${repo_dir}" fetch --depth "${REPO_DEPTH}" origin "${REPO_REF}"
        run_git -C "${repo_dir}" checkout --detach FETCH_HEAD
    else
        run_git -C "${repo_dir}" pull --ff-only
    fi
fi

run_git -C "${repo_dir}" submodule sync --recursive
run_git -C "${repo_dir}" submodule update \
    --init \
    --recursive \
    --depth "${REPO_DEPTH}" \
    -- "${ALPAMAYO_SUBMODULE_PATH}"

if [[ "${repo_dir}" != "${WORKSPACE_ROOT}" ]]; then
    for persistent_dir in data outputs; do
        source_dir="${WORKSPACE_ROOT}/${persistent_dir}"
        target_dir="${repo_dir}/${persistent_dir}"
        mkdir -p "${source_dir}"
        if [[ ! -e "${target_dir}" ]]; then
            ln -s "${source_dir}" "${target_dir}"
        fi
    done
fi

export PYTHONPATH="${repo_dir}:${repo_dir}/${ALPAMAYO_SUBMODULE_PATH}/src"

if [[ "${INSTALL_REPO_EDITABLE}" == "true" ]]; then
    python -m pip install --no-deps -e "${repo_dir}"
    python -m pip install --no-build-isolation --no-deps -e \
        "${repo_dir}/${ALPAMAYO_SUBMODULE_PATH}"
fi

cd "${repo_dir}"

if [[ "$#" -eq 0 ]]; then
    set -- bash
fi

exec "$@"
