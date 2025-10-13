#!/usr/bin/env bash
# Downloads one or more YouTube videos into data/video using the local yt-dlp.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$REPO_DIR/tools"
YT_DLP_BIN="$TOOLS_DIR/yt-dlp"
OUTPUT_DIR="$REPO_DIR/data/video"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <url1> [url2 ...]" >&2
    exit 1
fi

if [ ! -x "$YT_DLP_BIN" ]; then
    echo "Error: yt-dlp not found at $YT_DLP_BIN." >&2
    echo "Run ./install_youtube_tools.sh first." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

"$YT_DLP_BIN" -f b[ext=mp4] -o "$OUTPUT_DIR/%(id)s.%(ext)s" "$@"

echo "Downloads complete. Files saved in $OUTPUT_DIR"
