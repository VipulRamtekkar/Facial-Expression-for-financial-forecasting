#!/usr/bin/env bash
# macOS installer for facial-expression-analysis
# - Avoids compiling CPython (which fails on Apple Silicon with Python 3.6's
#   outdated configure), and instead uses the system/Homebrew Python 3.
# - Creates a local venv and installs pinned dependencies.
# - Includes notes for Apple Silicon (arm64) and optional Rosetta fallback.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv_env_emos"
REQUIREMENTS_FILE="$REPO_DIR/requirements-py36.txt"

command_exists() { command -v "$1" >/dev/null 2>&1; }

if [ "$(uname -s)" != "Darwin" ]; then
  echo "This script is for macOS only (Darwin detected via uname)." >&2
  exit 1
fi

arch="$(uname -m)"  # arm64 or x86_64
echo "==> Detected macOS architecture: ${arch}"

echo "==> Checking for Xcode Command Line Tools"
if ! xcode-select -p >/dev/null 2>&1; then
  echo "Xcode Command Line Tools not found. Install with: xcode-select --install" >&2
  exit 1
fi

# Prefer Homebrew Python if available; otherwise use system python3
PYTHON_CMD=${PYTHON_CMD:-python3}
if command_exists brew; then
  # Prefer <= 3.8 for best compatibility with pinned deps
  if brew list --versions python@3.8 >/dev/null 2>&1; then
    PYTHON_CMD="$(brew --prefix)/opt/python@3.8/bin/python3"
  elif brew list --versions python@3.9 >/dev/null 2>&1; then
    PYTHON_CMD="$(brew --prefix)/opt/python@3.9/bin/python3"
  elif brew list --versions python@3.10 >/dev/null 2>&1; then
    PYTHON_CMD="$(brew --prefix)/opt/python@3.10/bin/python3"
  elif brew list --versions python@3.11 >/dev/null 2>&1; then
    PYTHON_CMD="$(brew --prefix)/opt/python@3.11/bin/python3"
  elif brew list --versions python >/dev/null 2>&1; then
    # On some setups, python formula is alias to latest Python 3
    PYTHON_CMD="$(brew --prefix)/bin/python3"
  fi
fi

if ! command_exists "$PYTHON_CMD"; then
  echo "Error: python3 not found. Install Python 3 (e.g. 'brew install python')." >&2
  exit 1
fi

PY_VER=$("$PYTHON_CMD" -c 'import sys; print("%d.%d"%sys.version_info[:2])')
echo "==> Using Python ${PY_VER} at: $(command -v "$PYTHON_CMD")"
case "$PY_VER" in
  3.6|3.7|3.8) ;; # good
  *)
    echo "Warning: Detected Python ${PY_VER}. The pinned requirements were authored for Python <= 3.8." >&2
    echo "         If installation fails, install Python 3.8 and re-run, e.g.:" >&2
    echo "           brew install python@3.8" >&2
    echo "           PYTHON_CMD=\"\$(brew --prefix)/opt/python@3.8/bin/python3\" ./install_env_macos.sh" >&2
    ;;
esac

if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "Error: requirements file '$REQUIREMENTS_FILE' is missing." >&2
  exit 1
fi

echo "==> Preparing build prerequisites (cmake, pkg-config) if Homebrew exists"
if command_exists brew; then
  # We don't auto-install (network/offline), but we can nudge helpful guidance
  for pkg in cmake pkg-config; do
    if ! brew list --versions "$pkg" >/dev/null 2>&1; then
      echo "Hint: '$pkg' not found via Homebrew. If dlib build fails, run: brew install $pkg" >&2
    fi
  done
else
  echo "Hint: Homebrew not found. If dlib compilation fails, install cmake manually or install Homebrew from https://brew.sh." >&2
fi

# On Apple Silicon, many source-built wheels expect this target
if [ "$arch" = "arm64" ]; then
  export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-11.0}
  # Help native builds (e.g., dlib) find the correct SDK and target arch.
  if command_exists xcrun; then
    export SDKROOT=${SDKROOT:-"$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || true)"}
  fi
  # CMake hints for Apple Silicon
  export CMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES:-arm64}
  export CMAKE_ARGS="${CMAKE_ARGS:-} -DCMAKE_OSX_ARCHITECTURES=arm64"
  if command_exists brew; then
    export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-$(brew --prefix)}"
    # Allow pkg-config to find Homebrew libs
    export PKG_CONFIG_PATH="${PKG_CONFIG_PATH:-}:$(brew --prefix)/lib/pkgconfig:$(brew --prefix)/opt/openblas/lib/pkgconfig"
  fi
fi

mkdir -p "$REPO_DIR/.cache"

if [ ! -d "$VENV_DIR" ]; then
  echo "==> Creating virtual environment at $VENV_DIR"
  "$PYTHON_CMD" -m venv "$VENV_DIR"
else
  echo "==> Virtual environment already exists at $VENV_DIR"
fi

PIP_BIN="$VENV_DIR/bin/pip"
PY_BIN="$VENV_DIR/bin/python"

echo "==> Upgrading pip/setuptools/wheel"
"$PY_BIN" -m pip install --upgrade pip setuptools wheel

echo "==> Installing project requirements"
"$PIP_BIN" install --upgrade -r "$REQUIREMENTS_FILE"

ACTIVATE_PATH="$REPO_DIR/activate_env.sh"
# macOS doesn't ship realpath by default; use Python to resolve it nicely
if command -v python3 >/dev/null 2>&1; then
  ACTIVATE_PATH="$(python3 -c 'import os,sys;print(os.path.realpath(sys.argv[1]))' "$ACTIVATE_PATH")"
fi

cat <<EOF

Environment ready on macOS!

Notes:
- This script uses your existing Python 3 rather than compiling CPython 3.6.
  That avoids the classic "configure: error: unexpected arch for OSX" on M1.
- If dlib fails to build on Apple Silicon, ensure you have:
    brew install cmake pkg-config
  And try again. You may also set:
    export MACOSX_DEPLOYMENT_TARGET=11.0

Optional Rosetta fallback (Apple Silicon):
- If you must force x86_64 builds, install Rosetta and Homebrew x86_64, then
  run this script under Rosetta, e.g.:
    arch -x86_64 /bin/bash install_env_macos.sh
  But mixing architectures can complicate OpenCV/dlib builds, so prefer native arm64.

- Activate with:  source "$ACTIVATE_PATH"
- Python binary:  $PY_BIN
- Virtual env:    $VENV_DIR
EOF
