#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare baseline AR(1) vs ARX(1) with facial expression features
around a YouTube CEO earnings call.

Inputs:
- Video emotions CSV (from process_video_emotions.py)
- Upload datetime (ISO, with timezone if available)
- Stock ticker price data (minute bars). If not provided, script can fetch
  from Yahoo Finance if network is available.

Outputs:
- Summary metrics (MSE/MAE) for baseline vs exogenous model
- Plot of actual vs predictions saved under ../plots
"""

import argparse
import datetime as dt
import json
import io
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="DAL", help="Stock ticker symbol (default DAL)")
    p.add_argument(
        "--video-id",
        default="ywyBkewwcP0",
        help="YouTube video ID used to locate emotions CSV under data/video_emotions",
    )
    p.add_argument(
        "--emotions-csv",
        default=None,
        help="Optional explicit path to the emotions CSV. Defaults to data/video_emotions/<video-id>*.csv",
    )
    p.add_argument(
        "--upload-datetime",
        default=None,
        help=(
            "Upload datetime in ISO format (e.g. 2024-07-12T14:30:00-04:00). "
            "If not provided, the script looks for a JSON info file named data/video/<video-id>.info.json"
        ),
    )
    p.add_argument(
        "--market-csv",
        default=None,
        help=(
            "Optional path to a local CSV of minute OHLCV (Yahoo-style with a Date/Datetime column). "
            "If not provided, the script attempts to download from Yahoo Finance."
        ),
    )
    p.add_argument(
        "--output-prefix",
        default=None,
        help="Optional prefix for outputs saved under ../plots",
    )
    p.add_argument(
        "--use-intraday",
        action="store_true",
        help="If set, restrict to 09:30-16:00 ET; otherwise use full ±24h window.",
    )
    p.add_argument(
        "--arima-order",
        default="1,0,0",
        help="ARIMA order p,d,q (default 1,0,0).",
    )
    return p.parse_args()


def find_emotions_csv(video_id: str, explicit_path: Optional[str]) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    # default search
    base = Path(__file__).resolve().parents[1]
    cand = list((base / "data" / "video_emotions").glob(f"{video_id}*.csv"))
    if not cand:
        raise FileNotFoundError("Could not find emotions CSV; run process_video_emotions.py first.")
    # Prefer the non-sample if present
    cand_sorted = sorted(cand, key=lambda p: ("sample" in p.name, p.name))
    return cand_sorted[0]


def parse_upload_datetime(video_id: str, explicit_iso: Optional[str]) -> dt.datetime:
    if explicit_iso:
        return pd.Timestamp(explicit_iso).to_pydatetime()
    # fallback: look for info json
    base = Path(__file__).resolve().parents[1]
    info_path = base / "data" / "video" / f"{video_id}.info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            "Upload datetime not provided and info JSON not found. "
            "Supply --upload-datetime or save a yt-dlp JSON to data/video/<video-id>.info.json"
        )
    with open(info_path, "r") as f:
        info = json.load(f)
    # Try timestamp (seconds) else upload_date (YYYYMMDD)
    ts = info.get("timestamp") or info.get("release_timestamp")
    if ts:
        return dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc)
    upload_date = info.get("upload_date")
    if upload_date:
        # interpret as midnight UTC
        d = dt.datetime.strptime(upload_date, "%Y%m%d").date()
        return dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc)
    raise ValueError("Could not infer upload datetime from info JSON.")


def _ensure_utc(dt_in: dt.datetime) -> dt.datetime:
    if dt_in.tzinfo is None:
        return dt_in.replace(tzinfo=dt.timezone.utc)
    return dt_in.astimezone(dt.timezone.utc)


def yahoo_download(ticker: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    """Download 1m intraday data from Yahoo v8 chart API."""
    import urllib.request
    import json as _json

    start_utc = _ensure_utc(start)
    end_utc = _ensure_utc(end)
    p1 = int(start_utc.timestamp())
    p2 = int(end_utc.timestamp())

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={p1}&period2={p2}&interval=1m&includePrePost=true&events=div%2Csplit"
    )
    with urllib.request.urlopen(url) as resp:
        payload = resp.read().decode("utf-8")
    data = _json.loads(payload)
    result = data.get("chart", {}).get("result")
    if not result:
        raise RuntimeError("Yahoo chart API returned no result")
    result0 = result[0]
    ts = result0.get("timestamp", [])
    indicators = result0.get("indicators", {}).get("quote", [{}])[0]
    if not ts or "close" not in indicators:
        raise RuntimeError("Yahoo chart API missing timestamps or quotes")
    idx = pd.to_datetime(ts, unit="s", utc=True)
    df = pd.DataFrame(
        {
            "Open": indicators.get("open", [None] * len(ts)),
            "High": indicators.get("high", [None] * len(ts)),
            "Low": indicators.get("low", [None] * len(ts)),
            "Close": indicators.get("close", [None] * len(ts)),
            "Volume": indicators.get("volume", [None] * len(ts)),
        },
        index=idx,
    ).astype(float).sort_index()
    df = df.dropna(subset=["Close"])  # ensure valid bars
    return df


def load_market_data(
    ticker: str,
    upload_dt: dt.datetime,
    csv_path: Optional[str],
) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    window_start = pd.Timestamp(upload_dt - dt.timedelta(days=1))
    window_end = pd.Timestamp(upload_dt + dt.timedelta(days=1))

    if csv_path:
        df = pd.read_csv(csv_path)
        if ("Datetime" in df.columns) or ("Date" in df.columns):
            dt_col = "Datetime" if "Datetime" in df.columns else "Date"
            df[dt_col] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
            df = (
                df.dropna(subset=[dt_col])
                .rename(columns={dt_col: "Datetime"})
                .set_index("Datetime")
                .sort_index()
            )
        else:
            # assume first column is the datetime index
            df = pd.read_csv(csv_path, index_col=0)
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            df = df[~df.index.isna()].sort_index()
        # filter window in UTC
        start_utc = _ensure_utc(window_start.to_pydatetime())
        end_utc = _ensure_utc(window_end.to_pydatetime())
        df = df[(df.index >= pd.Timestamp(start_utc)) & (df.index <= pd.Timestamp(end_utc))]
        return df, window_start, window_end

    # Try Yahoo (network required)
    df = yahoo_download(ticker, window_start.to_pydatetime(), window_end.to_pydatetime())
    return df, window_start, window_end


def minute_intraday_filter(df: pd.DataFrame, tz: str = "America/New_York") -> pd.DataFrame:
    df = df.copy()
    df = df.tz_convert(tz)
    # keep regular session 09:30-16:00
    intraday = df.between_time("09:30", "16:00")
    return intraday


def build_returns(df: pd.DataFrame) -> pd.Series:
    close = df["Close"].astype(float)
    r = np.log(close).diff().dropna()
    return r


def prepare_exog_from_emotions(
    emotions_csv: Path,
    upload_dt: dt.datetime,
    idx_template: pd.DatetimeIndex,
    tz: str = "America/New_York",
) -> pd.DataFrame:
    emo = pd.read_csv(emotions_csv)
    # We only keep frames with detections
    emo = emo[emo.get("detected", True) == True]
    # absolute timestamps
    up = pd.Timestamp(upload_dt)
    if up.tz is None:
        up = up.tz_localize("UTC")
    else:
        up = up.tz_convert("UTC")
    up = up.tz_convert(tz)
    emo["abs_time"] = up + pd.to_timedelta(emo["timestamp_sec"], unit="s")
    emo["abs_minute"] = emo["abs_time"].dt.floor("T")
    agg = (
        emo.groupby("abs_minute")[
            ["arousal", "valence", "intensity"]
        ]
        .mean()
        .rename(columns={"arousal": "arousal_mean", "valence": "valence_mean", "intensity": "intensity_mean"})
    )
    # align to template index and fill missing with zeros (no emotion signal)
    exog = pd.DataFrame(index=idx_template)
    exog = exog.join(agg, how="left")
    exog = exog.fillna(0.0)
    return exog


def try_import_statsmodels():
    try:
        import statsmodels  # noqa: F401
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: F401
        return True
    except Exception:
        return False


def fit_arima_sm(train_r: pd.Series, order: Tuple[int, int, int]):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(train_r, order=order, trend="c", enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    return res


def fit_arimax_sm(train_r: pd.Series, train_x: pd.DataFrame, order: Tuple[int, int, int]):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    exog = train_x.loc[train_r.index]
    model = SARIMAX(train_r, exog=exog, order=order, trend="c", enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    return res


def fit_ar1(train_r: pd.Series) -> Tuple[float, float]:
    # r_t = c + phi * r_{t-1} + e_t
    y = train_r.iloc[1:].values
    x = train_r.shift(1).iloc[1:].values
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    c, phi = beta
    return float(c), float(phi)


def fit_arx1(train_r: pd.Series, train_x: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    # r_t = c + phi*r_{t-1} + b'*X_t + e_t
    y = train_r.iloc[1:].values
    rlag = train_r.shift(1).iloc[1:].values
    X = train_x.loc[train_r.index].iloc[1:].values
    ones = np.ones((len(y), 1))
    Z = np.column_stack([ones, rlag, X])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return beta, Z.mean(axis=0)  # return beta and column means for reference


def predict_ar1(c: float, phi: float, test_r: pd.Series) -> pd.Series:
    # One-step-ahead using actual r_{t-1}
    pred = c + phi * test_r.shift(1)
    return pred.dropna()


def predict_arx1(beta: np.ndarray, test_r: pd.Series, test_x: pd.DataFrame) -> pd.Series:
    # One-step-ahead using actual r_{t-1}
    aligned_x = test_x.loc[test_r.index]
    ones = pd.Series(1.0, index=test_r.index)
    # beta = [c, phi, b1, b2, ...]
    pred = beta[0] + beta[1] * test_r.shift(1)
    if aligned_x.shape[1] > 0:
        exog_part = (aligned_x * beta[2:]).sum(axis=1)
        pred = pred + exog_part
    return pred.dropna()


def mse(y_true: pd.Series, y_pred: pd.Series) -> float:
    df = pd.concat([y_true, y_pred], axis=1, keys=["y", "yhat"]).dropna()
    return float(((df["y"] - df["yhat"]) ** 2).mean())


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    df = pd.concat([y_true, y_pred], axis=1, keys=["y", "yhat"]).dropna()
    return float((df["y"] - df["yhat"]).abs().mean())


def main() -> None:
    args = parse_args()
    base = Path(__file__).resolve().parents[1]
    plots_dir = base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = args.output_prefix or f"{args.ticker}_{args.video_id}"

    emotions_csv = find_emotions_csv(args.video_id, args.emotions_csv)
    upload_dt = parse_upload_datetime(args.video_id, args.upload_datetime)

    # Load market data (minute)
    mkt, wstart, wend = load_market_data(args.ticker, upload_dt, args.market_csv)
    mkt = mkt.tz_localize("UTC") if mkt.index.tz is None else mkt
    if args.use_intraday:
        mkt = minute_intraday_filter(mkt)
    returns = build_returns(mkt)

    # Split exactly at upload timestamp: train = [upload-24h, upload), test = [upload, upload+24h]
    upload_ts = pd.Timestamp(upload_dt)
    if upload_ts.tz is None:
        upload_ts = upload_ts.tz_localize("UTC")
    else:
        upload_ts = upload_ts.tz_convert("UTC")
    start_utc = pd.Timestamp(_ensure_utc((pd.Timestamp(upload_dt) - pd.Timedelta(days=1)).to_pydatetime()))
    end_utc = pd.Timestamp(_ensure_utc((pd.Timestamp(upload_dt) + pd.Timedelta(days=1)).to_pydatetime()))
    train_r = returns[(returns.index >= start_utc) & (returns.index < upload_ts)]
    test_r = returns[(returns.index >= upload_ts) & (returns.index <= end_utc)]
    if len(train_r) < 10 or len(test_r) < 10:
        raise RuntimeError("Insufficient data in train/test after splitting by ±24h window. Consider disabling intraday filter or checking data availability.")

    # Build exogenous from emotions, aligned to the full intraday index (train+test)
    full_idx = returns.index
    exog = prepare_exog_from_emotions(emotions_csv, upload_dt, full_idx)

    # Parse ARIMA order
    try:
        order = tuple(int(x) for x in args.arima_order.split(","))
        assert len(order) == 3
    except Exception:
        order = (1, 0, 0)

    # Fit models (prefer statsmodels if available)
    used_sm = False
    if try_import_statsmodels():
        used_sm = True
        ar_res = fit_arima_sm(train_r, order=order)
        fc_ar = ar_res.get_forecast(steps=len(test_r))
        yhat_ar1 = pd.Series(fc_ar.predicted_mean.values, index=test_r.index)

        arx_res = fit_arimax_sm(train_r, exog, order=order)
        fc_arx = arx_res.get_forecast(steps=len(test_r), exog=exog.loc[test_r.index])
        yhat_arx = pd.Series(fc_arx.predicted_mean.values, index=test_r.index)
    else:
        # OLS fallback (ARX(1))
        c, phi = fit_ar1(train_r)
        beta, _ = fit_arx1(train_r, exog)
        yhat_ar1 = predict_ar1(c, phi, test_r)
        yhat_arx = predict_arx1(beta, test_r, exog)
    y_true = test_r.loc[yhat_ar1.index.intersection(yhat_arx.index)]
    yhat_ar1 = yhat_ar1.loc[y_true.index]
    yhat_arx = yhat_arx.loc[y_true.index]

    # Evaluate
    metrics = {
        "MSE_AR1": mse(y_true, yhat_ar1),
        "MAE_AR1": mae(y_true, yhat_ar1),
        "MSE_ARX": mse(y_true, yhat_arx),
        "MAE_ARX": mae(y_true, yhat_arx),
    }

    # Save summary
    summary_path = plots_dir / f"{out_prefix}_arimax_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Train window: [{start_utc.tz_convert('America/New_York')}, {upload_ts.tz_convert('America/New_York')})\n")
        f.write(f"Test window:  [{upload_ts.tz_convert('America/New_York')}, {end_utc.tz_convert('America/New_York')}]\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6e}\n")
        improv = (metrics["MSE_AR1"] - metrics["MSE_ARX"]) / metrics["MSE_AR1"] if metrics["MSE_AR1"] > 0 else np.nan
        f.write(f"MSE improvement (ARX over AR1): {improv:.2%}\n")
        f.write(f"Used statsmodels: {used_sm}\n")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    y_true.plot(ax=ax, label="Actual", color="k", linewidth=1)
    yhat_ar1.plot(ax=ax, label="AR(1) pred", alpha=0.8)
    yhat_arx.plot(ax=ax, label="ARX(1) + V/A pred", alpha=0.8)
    ax.set_title(f"{args.ticker} one-step returns: AR vs ARX with V/A")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{out_prefix}_arimax_compare.png", dpi=150)
    plt.close(fig)

    print(f"Saved summary to {summary_path}")
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6e}")


if __name__ == "__main__":
    main()
