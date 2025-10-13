#!/usr/bin/env bash
# Installs a standalone yt-dlp executable inside the repo for offline use.
# No sudo required. The binary is stored under tools/yt-dlp.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$REPO_DIR/tools"
YT_DLP_BIN="$TOOLS_DIR/yt-dlp"
YT_DLP_URL="https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp"

mkdir -p "$TOOLS_DIR"

echo "==> Downloading yt-dlp binary"
curl -L "$YT_DLP_URL" -o "$YT_DLP_BIN"
chmod +x "$YT_DLP_BIN"

cat <<EOF
yt-dlp installed at: $YT_DLP_BIN
Use ./download_youtube_videos.sh <urls...> to fetch videos.
EOF
