import argparse
from pathlib import Path

import cv2


def extract_frames(video_path, out_dir, t_start_s, t_end_s, stride_s):
    """
    Sample frames between t_start_s and t_end_s every stride_s seconds.
    Returns list of dicts with frame path and timestamp (seconds from video start).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(fps * stride_s))
    start_frame = int(fps * t_start_s)
    end_frame = int(fps * t_end_s)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    frame_idx = start_frame
    saved_idx = 0

    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % frame_interval == 0:
            frame_path = out_dir / f"frame_{saved_idx:05d}.png"
            cv2.imwrite(str(frame_path), frame)
            ts = frame_idx / fps
            saved.append({"frame": str(frame_path), "ts": ts})
            saved_idx += 1
        frame_idx += 1

    cap.release()
    return saved


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--t-start", type=float, default=0.0)
    parser.add_argument("--t-end", type=float, required=True)
    parser.add_argument("--stride", type=float, default=30.0)
    args = parser.parse_args()

    frames = extract_frames(
        args.video,
        args.out_dir,
        args.t_start,
        args.t_end,
        args.stride,
    )
    print(f"Saved {len(frames)} frames to {args.out_dir}")


if __name__ == "__main__":
    cli()
