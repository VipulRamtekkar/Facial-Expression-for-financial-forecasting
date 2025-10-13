#!/usr/bin/env bash
# Installs a self-contained Python 3.6 virtual environment for the
# facial-expression-analysis project without relying on Conda/Mamba.
# It compiles CPython locally and uses pip to install the required wheels.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="$REPO_DIR/.cache"
PYTHON_VERSION="3.6.15"
PYTHON_PREFIX="$REPO_DIR/.python-${PYTHON_VERSION}"
PYTHON_BIN="$PYTHON_PREFIX/bin/python3.6"
VENV_DIR="$REPO_DIR/.venv_env_emos"
BUILD_DIR="$REPO_DIR/.build-python-${PYTHON_VERSION}"
TARBALL="Python-${PYTHON_VERSION}.tgz"
PYTHON_URL="https://www.python.org/ftp/python/${PYTHON_VERSION}/${TARBALL}"
REQUIREMENTS_FILE="$REPO_DIR/requirements-py36.txt"

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "==> Preparing directories"
mkdir -p "$CACHE_DIR"

for tool in curl tar make gcc; do
    if ! command_exists "$tool"; then
        echo "Error: required build tool '$tool' not found in PATH." >&2
        echo "Please install the missing packages (e.g. sudo apt install build-essential curl tar)." >&2
        exit 1
    fi
done

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Error: requirements file '$REQUIREMENTS_FILE' is missing." >&2
    exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
    echo "==> Downloading CPython ${PYTHON_VERSION}"
    TARBALL_PATH="$CACHE_DIR/$TARBALL"
    if [ ! -f "$TARBALL_PATH" ]; then
        curl -L "$PYTHON_URL" -o "$TARBALL_PATH"
    else
        echo "    Using cached tarball $TARBALL_PATH"
    fi

    echo "==> Extracting sources"
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    tar -xzf "$TARBALL_PATH" -C "$BUILD_DIR"

    echo "==> Building CPython (this may take a while)"
    pushd "$BUILD_DIR/Python-${PYTHON_VERSION}" >/dev/null
    # Skip --enable-optimizations to avoid the lengthy PGO test suite during build
    ./configure --prefix="$PYTHON_PREFIX" --with-ensurepip=install
    make -j"$(nproc)"
    make install
    popd >/dev/null

    echo "==> Cleaning build directory"
    rm -rf "$BUILD_DIR"
else
    echo "==> Reusing existing CPython at $PYTHON_BIN"
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating virtual environment at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "==> Virtual environment already exists at $VENV_DIR"
fi

PIP_BIN="$VENV_DIR/bin/pip"
PY_BIN="$VENV_DIR/bin/python"

echo "==> Upgrading pip/setuptools/wheel"
"$PY_BIN" -m pip install --upgrade pip setuptools wheel

echo "==> Installing project requirements"
"$PIP_BIN" install --upgrade -r "$REQUIREMENTS_FILE"

cat <<EOF

Environment ready!
- Activate with:  source "$(realpath "$REPO_DIR/activate_env.sh")"
- Python binary:  $PY_BIN
- Virtual env:    $VENV_DIR
EOF
