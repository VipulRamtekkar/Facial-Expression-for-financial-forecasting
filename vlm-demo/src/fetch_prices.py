import argparse
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

DEFAULT_INTERVAL = "1m"


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_event_time(event_time: str) -> datetime:
    text = event_time.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return _ensure_utc(datetime.fromisoformat(text))


def _run_ffprobe(video_path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_entries",
        "format_tags=creation_time",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffprobe is not available. Install ffmpeg or pass --event-time."
        ) from exc
    if not result.stdout.strip():
        raise RuntimeError("ffprobe returned no output.")
    return json.loads(result.stdout)


def infer_publish_time(video_path: Path) -> datetime:
    data = _run_ffprobe(video_path)
    try:
        creation_time = data["format"]["tags"]["creation_time"]
    except KeyError as exc:
        raise RuntimeError(
            "creation_time metadata missing. Provide --event-time manually."
        ) from exc
    return parse_event_time(creation_time)


def download_prices(ticker: str, start_dt: datetime, end_dt: datetime, interval: str):
    data = yf.Ticker(ticker).history(
        interval=interval,
        start=start_dt,
        end=end_dt,
        actions=False,
    )
    if data.empty and interval == "1m":
        return download_prices(ticker, start_dt, end_dt, interval="5m")
    if data.empty:
        raise RuntimeError(
            f"No price data returned for {ticker} between {start_dt} and {end_dt}."
        )
    df = data.reset_index()
    time_col = "Datetime" if "Datetime" in df.columns else "Date"
    timestamps = pd.to_datetime(df[time_col], utc=True)
    # Persist UTC designator so downstream parsing keeps timezone awareness
    df["timestamp"] = timestamps.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    return df[cols], interval


def write_prices(df: pd.DataFrame, output_dir: Path, ticker: str, interval: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = interval.replace("m", "min")
    path = output_dir / f"{ticker.upper()}_{suffix}.csv"
    df.to_csv(path, index=False)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Fetch price data around a video's publish time and save to CSV."
    )
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. DAL.")
    parser.add_argument(
        "--event-time",
        help="Optional ISO timestamp (UTC preferred). "
        "Overrides metadata-derived publish time.",
    )
    parser.add_argument(
        "--window-before",
        type=int,
        default=240,
        help="Minutes before event time to include (default: 240).",
    )
    parser.add_argument(
        "--window-after",
        type=int,
        default=240,
        help="Minutes after event time to include (default: 240).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/prices",
        help="Directory for output CSV (default: data/prices).",
    )
    parser.add_argument(
        "--interval",
        default=DEFAULT_INTERVAL,
        help="Yahoo Finance interval (default 1m; falls back to 5m automatically).",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if args.event_time:
        event_time = parse_event_time(args.event_time)
    else:
        event_time = infer_publish_time(video_path)

    start_dt = _ensure_utc(event_time - timedelta(minutes=args.window_before))
    end_dt = _ensure_utc(event_time + timedelta(minutes=args.window_after))

    prices, interval_used = download_prices(
        args.ticker, start_dt=start_dt, end_dt=end_dt, interval=args.interval
    )

    output_path = write_prices(
        prices, Path(args.output_dir), args.ticker, interval_used
    )

    print(f"Wrote {len(prices)} rows to {output_path}")
    print(f"Event time (UTC): {event_time.isoformat()}")
    if interval_used != args.interval:
        print(f"Used fallback interval: {interval_used}")


if __name__ == "__main__":
    main()
