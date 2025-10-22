from datetime import timezone

import pandas as pd


def _ensure_utc(dt):
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_price_features(csv_path, event_timestamp, horizons):
    bars = pd.read_csv(csv_path, parse_dates=["timestamp"])
    bars = bars.sort_values("timestamp").set_index("timestamp")

    event_ts = _ensure_utc(event_timestamp)
    if bars.index.tz is None:
        event_ts = event_ts.astimezone(timezone.utc).replace(tzinfo=None)
    else:
        event_ts = event_ts.astimezone(bars.index.tz)

    if event_ts not in bars.index:
        bars = bars.reindex(bars.index.union([event_ts])).sort_index()
        bars = bars.interpolate(method="time").ffill().bfill()

    current = bars.loc[event_ts]

    features = {
        "px_close": current["close"],
        "px_ret_5m": bars["close"].pct_change(5).loc[event_ts],
        "px_vol_norm": (
            bars["volume"].rolling(30, min_periods=1).mean() / bars["volume"]
        ).loc[event_ts],
    }

    labels = {}
    for horizon in horizons:
        future_ts = event_ts + pd.Timedelta(minutes=horizon)
        if future_ts not in bars.index:
            continue
        future_ret = (bars["close"].loc[future_ts] / current["close"]) - 1.0
        labels[f"ret_{horizon}m"] = future_ret
        labels[f"dir_{horizon}m"] = int(future_ret > 0)

    return features, labels
