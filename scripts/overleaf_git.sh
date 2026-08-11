#!/usr/bin/env bash

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
env_file="${repo_root}/.env"

if [[ ! -f "${env_file}" ]]; then
  echo "Missing ${env_file}; copy .env.example and add the private token." >&2
  exit 1
fi

set -a
source "${env_file}"
set +a

: "${OVERLEAF_GIT_URL:?Missing OVERLEAF_GIT_URL in .env}"
: "${OVERLEAF_GIT_USERNAME:?Missing OVERLEAF_GIT_USERNAME in .env}"
: "${OVERLEAF_GIT_TOKEN:?Missing OVERLEAF_GIT_TOKEN in .env}"
: "${OVERLEAF_PROJECT_PATH:?Missing OVERLEAF_PROJECT_PATH in .env}"

submodule_name="paper/manuscript"
project_root="${repo_root}/${OVERLEAF_PROJECT_PATH}"
askpass_file="$(mktemp)"

cleanup() {
  unset OVERLEAF_GIT_TOKEN
  unlink "${askpass_file}" 2>/dev/null || true
}
trap cleanup EXIT

cat >"${askpass_file}" <<'EOF'
#!/bin/sh
case "$1" in
  *Username*) printf '%s\n' "$OVERLEAF_GIT_USERNAME" ;;
  *Password*) printf '%s\n' "$OVERLEAF_GIT_TOKEN" ;;
esac
EOF
chmod 700 "${askpass_file}"

run_authenticated() {
  GIT_ASKPASS="${askpass_file}" GIT_TERMINAL_PROMPT=0 "$@"
}

usage() {
  cat <<'EOF'
Usage: scripts/overleaf_git.sh <command> [git arguments]

Commands:
  init       Initialize or restore the paper/manuscript submodule
  status     Show the manuscript worktree status
  fetch      Fetch the Overleaf remote
  pull       Fast-forward the current manuscript branch
  push       Push the current manuscript branch
  git ...    Run an authenticated Git command inside the manuscript repo
EOF
}

case "${1:-}" in
  init)
    git -C "${repo_root}" config "submodule.${submodule_name}.url" "${OVERLEAF_GIT_URL}"
    run_authenticated git -C "${repo_root}" submodule update --init --recursive -- "${OVERLEAF_PROJECT_PATH}"
    ;;
  status)
    git -C "${project_root}" status --short --branch
    ;;
  fetch)
    run_authenticated git -C "${project_root}" fetch --prune origin
    ;;
  pull)
    run_authenticated git -C "${project_root}" pull --ff-only
    ;;
  push)
    run_authenticated git -C "${project_root}" push
    ;;
  git)
    shift
    run_authenticated git -C "${project_root}" "$@"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
