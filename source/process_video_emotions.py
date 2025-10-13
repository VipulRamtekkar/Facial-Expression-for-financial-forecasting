#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process a video to extract per-frame facial expression estimates and save them
as a CSV file that can be consumed by downstream analyses.
"""

import argparse
import math
from pathlib import Path

import cv2
import dlib
import numpy as np
import pandas as pd

from emotions_dlib import EmotionsDlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract arousal/valence/intensity estimates from a video."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the output CSV file with per-frame estimates.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory that contains the pretrained models. "
        "Defaults to <repo_root>/models.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Use every Nth frame to speed up processing (default: 1).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional hard limit on the number of frames to process.",
    )
    parser.add_argument(
        "--drop-missing",
        action="store_true",
        help="If set, rows without a detected face are excluded from the CSV.",
    )
    return parser.parse_args()


def _select_primary_face(faces) -> int:
    """Return index of the largest detected face."""
    if not faces:
        return -1
    if len(faces) == 1:
        return 0
    areas = [
        (face.bottom() - face.top()) * (face.right() - face.left())
        for face in faces
    ]
    return int(np.argmax(areas))


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    repo_root = Path(__file__).resolve().parents[1]
    model_dir = Path(args.model_dir).expanduser().resolve() if args.model_dir else repo_root / "models"
    predictor_path = model_dir / "shape_predictor_68_face_landmarks.dat"
    emotion_model_path = next(model_dir.glob("model_emotion_*.joblib"), None)
    frontal_model_path = model_dir / "model_frontalization.npy"

    if not predictor_path.exists():
        raise FileNotFoundError(f"Could not locate DLIB shape predictor at {predictor_path}")
    if emotion_model_path is None or not emotion_model_path.exists():
        raise FileNotFoundError("Could not locate the pretrained emotion model (*.joblib).")
    if not frontal_model_path.exists():
        raise FileNotFoundError(f"Could not locate frontalization weights at {frontal_model_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0:
        fps = 30.0  # fallback when FPS metadata is missing

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(predictor_path))
    emotion_estimator = EmotionsDlib(
        file_emotion_model=str(emotion_model_path),
        file_frontalization_model=str(frontal_model_path),
    )

    rows = []
    frame_idx = 0
    processed = 0
    max_frames = args.max_frames if args.max_frames is None or args.max_frames > 0 else None

    while True:
        if max_frames is not None and processed >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.frame_step != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        row = {
            "frame_index": frame_idx,
            "timestamp_sec": frame_idx / fps,
            "detected": False,
            "arousal": np.nan,
            "valence": np.nan,
            "intensity": np.nan,
            "emotion_label": None,
        }

        primary_idx = _select_primary_face(faces)
        if primary_idx >= 0:
            landmarks_object = predictor(gray, faces[primary_idx])
            dict_emotions = emotion_estimator.get_emotions(landmarks_object)
            emotions = dict_emotions["emotions"]

            row.update(
                {
                    "detected": True,
                    "arousal": emotions["arousal"],
                    "valence": emotions["valence"],
                    "intensity": emotions["intensity"],
                    "emotion_label": emotions["name"],
                }
            )

        rows.append(row)
        processed += 1
        frame_idx += 1

    cap.release()

    df = pd.DataFrame(rows)
    if args.drop_missing:
        df = df[df["detected"]].reset_index(drop=True)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")

    if df["detected"].any():
        detected_df = df[df["detected"]]
        summary = detected_df[["arousal", "valence", "intensity"]].describe()
        print("Summary statistics for detected frames:")
        print(summary)
    else:
        print("Warning: no faces detected in the processed frames.")


if __name__ == "__main__":
    main()
