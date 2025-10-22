import pandas as pd


def build_vlm_features(scored_rows, window_s=60, min_conf=0.5):
    if not scored_rows:
        raise ValueError("No VLM rows provided.")

    df = pd.DataFrame(scored_rows)
    df = df[df["confidence"] >= min_conf].copy()
    if df.empty:
        raise ValueError("All rows filtered out by confidence threshold.")

    df["bin"] = (df["ts"] // window_s).astype(int)
    agg = df.groupby("bin").agg(
        v_mean=("valence", "mean"),
        v_std=("valence", "std"),
        a_mean=("arousal", "mean"),
        i_mean=("intensity", "mean"),
        i_max=("intensity", "max"),
        n_frames=("valence", "size"),
    ).fillna(0)

    for lag in (1, 2, 3):
        for col in ("v_mean", "a_mean", "i_mean"):
            key = f"{col}_lag{lag}"
            agg[key] = agg[col].shift(lag).fillna(agg[col].iloc[0])

    return agg.reset_index(drop=True)
