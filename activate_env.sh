#!/usr/bin/env bash
# Activates the locally built Python virtual environment for the project.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv_env_emos"

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: virtual environment not found at $VENV_DIR." >&2
    echo "Please run ./install_env.sh first." >&2
    exit 1
fi

echo "Activating environment from $VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

export PYTHONPATH="$REPO_DIR/source:${PYTHONPATH:-}"

echo "Environment activated. Python: $(command -v python)"
