import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from extract_frames import extract_frames
from openrouter_vlm import score_frame
from features_vlm import build_vlm_features
from features_price import load_price_features
from train_eval import train_directional_model


logger = logging.getLogger(__name__)


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def run_demo(args):
    Path(args.frames_dir).mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting demo run for %s | video=%s | event=%s | horizons=%s",
        args.ticker,
        args.video,
        args.event_time,
        args.horizons,
    )

    frames = extract_frames(
        args.video,
        args.frames_dir,
        args.t_start,
        args.t_end,
        args.frame_stride,
    )
    logger.info("Extracted %d frames into %s", len(frames), args.frames_dir)
    if not frames:
        raise RuntimeError("Frame extraction yielded no frames. Check timestamps.")

    scored = []
    total_frames = len(frames)
    logger.info("Scoring %d frames with VLM model...", total_frames)
    for idx, row in enumerate(frames, start=1):
        payload = score_frame(row["frame"])
        logger.debug("Scored frame %s/%s | ts=%.2f", idx, total_frames, row["ts"])
        scored.append({**row, **payload})
    logger.info("Completed VLM scoring for %d frames", total_frames)

    feature_df = build_vlm_features(
        scored,
        window_s=args.window,
        min_conf=args.min_conf,
    )
    logger.info(
        "Constructed VLM feature matrix with %d rows and %d columns",
        feature_df.shape[0],
        feature_df.shape[1],
    )

    event_ts = _ensure_utc(datetime.fromisoformat(args.event_time))
    px_features, labels = load_price_features(
        args.bars,
        event_ts,
        args.horizons,
    )
    logger.info(
        "Loaded price features aligned to %s | available labels=%s",
        event_ts.isoformat(),
        sorted(labels.keys()),
    )

    perf_features = feature_df.assign(**px_features)
    logger.debug("Combined feature dataframe shape: %s", perf_features.shape)

    for horizon in args.horizons:
        label_key = f"dir_{horizon}m"
        if label_key not in labels:
            print(f"Skipping {horizon}m horizon: missing price label.")
            continue

        label_series = pd.Series([labels[label_key]] * len(perf_features))
        model, score = train_directional_model(perf_features, label_series)

        if model is None:
            direction = "↑" if labels[label_key] else "↓"
            print(
                f"{args.ticker} | {horizon}m -> {direction} "
                "(insufficient samples for model training)"
            )
            continue

        probs = model.predict_proba(perf_features.values)[-1, 1]
        direction = "↑" if probs >= 0.5 else "↓"
        metric = f"{score:.2f}" if score is not None else "n/a"
        print(
            f"{args.ticker} | {horizon}m -> {direction} "
            f"(p={probs:.2f}) | holdout acc={metric}"
        )
        logger.info(
            "Finished horizon %sm | probability=%.3f | holdout= %s",
            horizon,
            probs,
            metric,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames-dir", default="data/frames")
    parser.add_argument("--bars", required=True)
    parser.add_argument("--event-time", required=True)
    parser.add_argument("--t-start", type=float, default=0.0)
    parser.add_argument("--t-end", type=float, required=True)
    parser.add_argument("--frame-stride", type=float, default=30.0)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--min-conf", type=float, default=0.5)
    parser.add_argument(
        "--horizons",
        "--horizon",
        nargs="+",
        type=int,
        default=[5, 60],
        dest="horizons",
        help="Prediction horizons in minutes (accepts multiple values).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    run_demo(args)


if __name__ == "__main__":
    main()
