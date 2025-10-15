
#!/usr/bin/env python3
"""
Face Signal Experiment
======================

Goal: Test whether facial-expression features (valence/arousal/intensity) measured during an earnings call
provide *incremental* predictive power for short-horizon stock direction, beyond simple baselines.

Inputs (CSV files)
------------------
Required:
  1) faces_timeseries.csv
     Columns: event_id, ticker, ts, valence, arousal, intensity
     - ts must be parseable datetime (UTC ok).
     - Multiple rows per event_id (e.g., 2–5 Hz sampling).

  2) events.csv
     Columns: event_id, ticker, call_date
     - call_date in YYYY-MM-DD (date of the call).

  3) prices.csv
     Columns: date, ticker, close
     - Daily closes spanning the window for your events.

Optional:
  4) controls_event.csv
     Columns: event_id, (any control features, e.g. eps_surprise, text_sentiment, audio_energy, etc.)

  5) vlm_event_embeddings.csv
     Columns: event_id, vlm_0 ... vlm_N (or a few prompt-based scores named vlm_*).

Outputs
-------
- results/summary.txt                      : Key metrics and deltas
- results/preds_event_level.csv            : Per-event probabilities and directions for event-level models
- results/preds_arima_arimax.csv           : ARIMA / ARIMAX per-event probabilities and directions
- results/local_projections_summary.csv    : Coefs and p-values (faces) for h=1..3 days (in-sample & oos)

Usage
-----
$ pip install pandas numpy scikit-learn statsmodels scipy
$ python face_signal_experiment.py --data-dir /path/to/csvs --test-split 0.2 --embargo-days 3 --arimax-decay 1.0,0.5,0.25

Notes
-----
- This is a compact, honest smoke-test. Start here; if faces add +2–5% MDA out-of-time consistently, iterate.
"""

import argparse
import os
import sys
import json
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.utils import resample
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

warnings.filterwarnings("ignore")


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def mda(y_true, y_dir):
    return float(accuracy_score(y_true, y_dir))


def sign_prob_to_dir(p):
    return (np.asarray(p) >= 0.5).astype(int)


def brier(y_true, p):
    return float(brier_score_loss(y_true, p))


def safe_auc(y_true, p):
    u = np.unique(y_true)
    if len(u) < 2:
        return np.nan
    return float(roc_auc_score(y_true, p))


def parse_decay(s: str) -> List[float]:
    return [float(x) for x in s.split(",")] if s else [1.0, 0.5, 0.25]


def time_split_by_fraction(df: pd.DataFrame, time_col: str, test_frac: float, embargo_days: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split with optional embargo gap between train and test."""
    df = df.sort_values(time_col).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_frac))
    if split_idx <= 0:
        split_idx = max(len(df) - 1, 1)
    cutoff_date = df.loc[split_idx - 1, time_col]
    # apply embargo: remove a buffer around cutoff from both sides
    if embargo_days > 0:
        emb_lo = cutoff_date - pd.Timedelta(days=embargo_days)
        emb_hi = cutoff_date + pd.Timedelta(days=embargo_days)
        train = df[df[time_col] <= emb_lo]
        test = df[df[time_col] >= emb_hi]
    else:
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
    return train, test


# -----------------------------
# Feature engineering: faces -> event-level vector
# -----------------------------
def _slope(y: np.ndarray) -> float:
    if len(y) < 2: return 0.0
    x = np.arange(len(y))
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return float(np.dot(x, y) / (len(y)-1))


def _madiff(y: np.ndarray) -> float:
    if len(y) < 2: return 0.0
    return float(np.mean(np.abs(np.diff(y))))


def _share_above(y: np.ndarray, thr: float = 0.2) -> float:
    if len(y) == 0: return 0.0
    return float(np.mean(y > thr))


def build_face_event_features(faces_df: pd.DataFrame) -> pd.DataFrame:
    req = {"event_id", "valence", "arousal", "intensity"}
    missing = req - set(faces_df.columns)
    if missing:
        raise ValueError(f"faces_timeseries.csv missing columns: {missing}")
    feats = []
    for eid, g in faces_df.groupby("event_id"):
        g = g.sort_values("ts")
        row = {"event_id": eid}
        for name, short in [("valence","val"), ("arousal","aro"), ("intensity","int")]:
            s = pd.to_numeric(g[name], errors="coerce").dropna().values
            if len(s) == 0:
                s = np.array([0.0])
            row[f"{short}_mean"]   = float(np.mean(s))
            row[f"{short}_std"]    = float(np.std(s))
            row[f"{short}_slope"]  = _slope(s)
            row[f"{short}_madiff"] = _madiff(s)
            row[f"{short}_p20"]    = _share_above(s, 0.2)
        feats.append(row)
    return pd.DataFrame(feats)


# -----------------------------
# Controls from prices
# -----------------------------
def add_price_controls(events_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """Adds prior 5D / 20D momentum and 20D volatility as controls per event."""
    px = prices_df.sort_values(["ticker","date"]).copy()
    px["logp"] = np.log(px["close"])
    px["ret"] = px.groupby("ticker")["logp"].diff()
    px["ret_sq"] = px["ret"]**2
    px["mom5"]  = px.groupby("ticker")["logp"].diff(5)
    px["mom20"] = px.groupby("ticker")["logp"].diff(20)
    px["vol20"] = (px.groupby("ticker")["ret_sq"].rolling(20).mean().reset_index(level=0, drop=True))**0.5
    # Merge nearest prior day features for each event
    merged = pd.merge_asof(
        events_df.sort_values("call_date"),
        px.sort_values("date"),
        left_on="call_date",
        right_on="date",
        by="ticker",
        direction="backward"
    )
    out = events_df.merge(merged[["event_id","mom5","mom20","vol20"]], on="event_id", how="left")
    return out


# -----------------------------
# Labels from prices
# -----------------------------
def build_labels_nextday(events_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """Label = 1 if next trading day's log-return > 0; also include numeric ret for AUC/Brier."""
    px = prices_df.sort_values(["ticker","date"]).copy()
    px["ret_next"] = px.groupby("ticker")["close"].shift(-1) / px["close"]
    px["ret_next"] = np.log(px["ret_next"])
    labels = px.rename(columns={"date": "call_date"})[["ticker","call_date","ret_next"]].copy()
    labels["direction"] = (labels["ret_next"] > 0).astype(int)
    return events_df.merge(labels, on=["ticker","call_date"], how="left").dropna(subset=["direction"])


# -----------------------------
# Event-level models
# -----------------------------
def run_event_level_models(
    events: pd.DataFrame,
    face_feats: pd.DataFrame,
    controls_event: Optional[pd.DataFrame],
    vlm: Optional[pd.DataFrame],
    test_frac: float,
    embargo_days: int
) -> Dict:
    df = events.merge(face_feats, on="event_id", how="left")
    if controls_event is not None and len(controls_event.columns) > 1:
        df = df.merge(controls_event, on="event_id", how="left")
    if vlm is not None and len(vlm.columns) > 1:
        df = df.merge(vlm, on="event_id", how="left")

    # Feature sets
    face_cols = [c for c in df.columns if c.startswith(("val_","aro_","int_"))]
    ctrl_cols = [c for c in df.columns if c not in set(["event_id","ticker","call_date","ret_next","direction"]) | set(face_cols)]
    # Keep practical control subset: price-derived controls + any user controls starting with ctrl_ or known names
    price_ctrls = [c for c in ["mom5","mom20","vol20"] if c in df.columns]
    user_ctrls  = [c for c in df.columns if c.startswith("ctrl_")]
    strong_ctrls = price_ctrls + user_ctrls

    # Build VLM branch if present
    vlm_cols = [c for c in df.columns if c.startswith("vlm_")]

    # Split
    data = df.dropna(subset=["direction"]).sort_values("call_date").reset_index(drop=True)
    # Ensure numeric feature blocks do not contain NaNs/Infs before feeding into sklearn
    feature_cols = list(dict.fromkeys(strong_ctrls + face_cols + vlm_cols))
    if feature_cols:
        data[feature_cols] = (
            data[feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
    if len(data) < 20:
        print("[WARN] Very small sample; results may be unstable.")
    train, test = time_split_by_fraction(data, "call_date", test_frac, embargo_days)

    y_tr, y_te = train["direction"].values, test["direction"].values

    # Handle degenerate class scenario (e.g., all ones) to avoid sklearn errors
    if len(np.unique(y_tr)) < 2:
        print("[WARN] Train split has <2 outcome classes; using constant baseline predictions.")
        base_prob = float(y_tr.mean()) if len(y_tr) else 0.5
        p_ctrl = np.full(len(y_te), base_prob)
        d_ctrl = sign_prob_to_dir(p_ctrl)
        p_cf = p_ctrl.copy()
        d_cf = d_ctrl.copy()
        p_cfv = p_ctrl.copy() if vlm_cols else None
        d_cfv = d_ctrl.copy() if vlm_cols else None
        # Assemble predictions DataFrame even if empty
        preds = test[["event_id","ticker","call_date","direction","ret_next"]].copy()
        preds["p_ctrl"] = p_ctrl
        preds["d_ctrl"] = d_ctrl
        preds["p_cf"] = p_cf
        preds["d_cf"] = d_cf
        if vlm_cols:
            preds["p_cfv"] = p_cfv
            preds["d_cfv"] = d_cfv

        def metrics(y, p, d):
            if len(y) == 0:
                return {"MDA": np.nan, "AUC": np.nan, "Brier": np.nan}
            return {
                "MDA": mda(y, d),
                "AUC": safe_auc(y, p),
                "Brier": brier(y, p)
            }

        met_ctrl = metrics(y_te, p_ctrl, d_ctrl)
        met_cf = metrics(y_te, p_cf, d_cf)
        met_cfv = metrics(y_te, p_cfv, d_cfv) if vlm_cols else None

        return {
            "metrics": {"CTRL": met_ctrl, "CTRL+FACE": met_cf, "CTRL+FACE+VLM": met_cfv},
            "mcnemar_p": None,
            "preds": preds
        }

    # Pipelines
    def make_pipe(use_faces: bool, use_vlm: bool):
        branches = []
        if strong_ctrls:
            branches.append(("ctrl", StandardScaler(with_mean=True, with_std=True), strong_ctrls))
        if use_faces and face_cols:
            branches.append(("face", StandardScaler(), face_cols))
        if use_vlm and vlm_cols:
            k = min(16, len(vlm_cols))
            branches.append(("vlm", Pipeline([("scaler", StandardScaler()),
                                              ("pca", PCA(n_components=k))]), vlm_cols))
        pre = ColumnTransformer(transformers=branches, remainder="drop")
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        return Pipeline([("pre", pre), ("clf", clf)])

    # Baseline: controls only
    pipe_ctrl = make_pipe(use_faces=False, use_vlm=False)
    pipe_ctrl.fit(train, y_tr)
    p_ctrl = pipe_ctrl.predict_proba(test)[:,1]
    d_ctrl = sign_prob_to_dir(p_ctrl)

    # Controls + faces
    pipe_cf = make_pipe(use_faces=True, use_vlm=False)
    pipe_cf.fit(train, y_tr)
    p_cf = pipe_cf.predict_proba(test)[:,1]
    d_cf = sign_prob_to_dir(p_cf)

    # Controls + faces + VLM (if available)
    p_cfv = None; d_cfv = None
    if vlm_cols:
        pipe_cfv = make_pipe(use_faces=True, use_vlm=True)
        pipe_cfv.fit(train, y_tr)
        p_cfv = pipe_cfv.predict_proba(test)[:,1]
        d_cfv = sign_prob_to_dir(p_cfv)

    # Metrics
    def metrics(y, p, d):
        return {
            "MDA": mda(y, d),
            "AUC": safe_auc(y, p),
            "Brier": brier(y, p)
        }

    met_ctrl = metrics(y_te, p_ctrl, d_ctrl)
    met_cf   = metrics(y_te, p_cf,   d_cf)
    met_cfv  = metrics(y_te, p_cfv,  d_cfv) if p_cfv is not None else None

    # McNemar test for direction difference (controls vs controls+faces)
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        tbl = np.zeros((2,2), dtype=int)
        # rows: ctrl correct/incorrect; cols: cf correct/incorrect
        ctrl_correct = (d_ctrl == y_te)
        cf_correct   = (d_cf   == y_te)
        for i in range(len(y_te)):
            tbl[0 if ctrl_correct[i] else 1, 0 if cf_correct[i] else 1] += 1
        mcnemar_p = mcnemar(tbl, exact=True).pvalue
    except Exception:
        mcnemar_p = np.nan

    preds = test[["event_id","ticker","call_date","direction","ret_next"]].copy()
    preds["p_ctrl"] = p_ctrl; preds["d_ctrl"] = d_ctrl
    preds["p_cf"]   = p_cf;   preds["d_cf"]   = d_cf
    if p_cfv is not None:
        preds["p_cfv"] = p_cfv; preds["d_cfv"] = d_cfv

    return {
        "metrics": {"CTRL": met_ctrl, "CTRL+FACE": met_cf, "CTRL+FACE+VLM": met_cfv},
        "mcnemar_p": float(mcnemar_p) if mcnemar_p==mcnemar_p else None,
        "preds": preds
    }


# -----------------------------
# Local Projections (h = 1..3 days)
# -----------------------------
def run_local_projections(
    events_with_feats: pd.DataFrame,
    prices_df: pd.DataFrame,
    face_cols: List[str],
    ctrl_cols: List[str],
    horizons: List[int],
    test_frac: float,
    embargo_days: int
) -> pd.DataFrame:
    """Regress future h-day returns on controls + faces measured at event time.
       Returns coefficients and p-values for faces, and a simple OOS directional metric by horizon.
    """
    # Build future returns per event for each horizon
    px = prices_df.sort_values(["ticker","date"]).copy()
    px["logp"] = np.log(px["close"])

    # Map call_date to index and compute forward returns
    events_sorted = events_with_feats.sort_values("call_date").reset_index(drop=True)
    out_rows = []

    # simple time split
    tr_events, te_events = time_split_by_fraction(events_sorted, "call_date", test_frac, embargo_days)

    for h in horizons:
        def future_ret(ticker, date):
            # r_{t+h} = log P_{t+h} - log P_t (close-to-close over h days)
            series = px[px["ticker"]==ticker].sort_values("date")[["date","logp"]].reset_index(drop=True)
            # find index of date
            idx = series.index[series["date"]==date]
            if len(idx)==0: return np.nan
            i = idx[0]
            j = i + h
            if j >= len(series): return np.nan
            return float(series.loc[j, "logp"] - series.loc[i, "logp"])

        for split_label, evs in [("train", tr_events), ("test", te_events)]:
            X = evs[ctrl_cols + face_cols].copy()
            X = X.replace([np.inf,-np.inf], np.nan).fillna(0.0)
            y = [future_ret(tkr, dt) for tkr, dt in zip(evs["ticker"], evs["call_date"])]
            y = pd.Series(y, index=evs.index).dropna()
            X = X.loc[y.index]

            if len(X) < 10 or len(X.columns)==0:
                continue

            X_const = sm.add_constant(X.values)
            model = sm.OLS(y.values, X_const).fit()

            # face block t-stats (use an index on concatenated cols)
            face_idx = [i for i, c in enumerate(["const"] + ctrl_cols + face_cols) if c in face_cols]
            # Save aggregate info
            # Also compute a crude directional oos metric for test split
            dir_acc = np.nan
            if split_label == "test":
                # direction from predicted sign
                yhat = model.predict(sm.add_constant(X.values))
                dir_acc = accuracy_score((y.values>0).astype(int), (yhat>0).astype(int))

            out_rows.append({
                "split": split_label,
                "h_days": h,
                "n_obs": int(len(y)),
                "oos_dir_acc": float(dir_acc) if dir_acc==dir_acc else np.nan,
                # Store mean coefficient magnitude over face cols & min p-value in face block
                "faces_coef_L2": float(np.linalg.norm(model.params[face_idx])) if face_idx else np.nan,
                "faces_min_p": float(np.min(model.pvalues[face_idx])) if face_idx else np.nan
            })

    return pd.DataFrame(out_rows)


# -----------------------------
# ARIMA vs ARIMAX with face impulses
# -----------------------------
def build_face_impulses(
    events_with_face_feats: pd.DataFrame,
    prices_df: pd.DataFrame,
    decay: List[float]
) -> pd.DataFrame:
    """Create daily exogenous variables per ticker:
       - Reduce face features to a single FACE_PC1 via PCA (on entire set for simplicity of construction)
       - On call_date set FACE_IMP_h0 = PC1, next days decay by provided weights.
    """
    # Reduce face features
    face_cols = [c for c in events_with_face_feats.columns if c.startswith(("val_","aro_","int_"))]
    if not face_cols:
        raise ValueError("No face feature columns found to build impulses.")
    X = events_with_face_feats[face_cols].fillna(0.0).values
    pc = PCA(n_components=1).fit_transform(StandardScaler().fit_transform(X)).flatten()
    ev_pc = events_with_face_feats[["event_id","ticker","call_date"]].copy()
    ev_pc["face_pc1"] = pc

    # Build calendar of exogs per ticker/date
    px = prices_df.sort_values(["ticker","date"]).copy()
    out = []
    for (tkr), g in px.groupby("ticker"):
        g = g[["date"]].copy()
        g["ticker"] = tkr
        for h, w in enumerate(decay):
            g[f"FACE_IMP_h{h}"] = 0.0
        # mark event impulses
        evs = ev_pc[ev_pc["ticker"]==tkr]
        date_to_imp = {pd.Timestamp(d): v for d, v in zip(evs["call_date"], evs["face_pc1"])}
        for i, row in g.iterrows():
            d = row["date"]
            if d in date_to_imp:
                v = date_to_imp[d]
                # current day and decays
                for h, w in enumerate(decay):
                    idx = g.index.get_loc(i) if isinstance(i, (int, np.integer)) else i
                    j = idx + h
                    if j < len(g):
                        g.loc[g.index[j], f"FACE_IMP_h{h}"] = v * w
        out.append(g)
    exog = pd.concat(out, ignore_index=True)
    return exog


def run_arima_arimax(
    prices_df: pd.DataFrame,
    events_df: pd.DataFrame,
    face_impulses: pd.DataFrame,
    test_frac: float
) -> pd.DataFrame:
    """For each ticker, fit ARIMA(1,0,0) and ARIMAX(1,0,0)+FACE_IMP_* on train period.
       Forecast the next-day return for event dates in test period; convert to up-prob.
    """
    results = []
    # define train/test by calendar time (same cut for all tickers)
    all_dates = sorted(prices_df["date"].unique())
    cut_idx = int(len(all_dates) * (1 - test_frac))
    cut_idx = max(min(cut_idx, len(all_dates)-2), 1)
    cutoff = all_dates[cut_idx]

    for tkr, g in prices_df.groupby("ticker"):
        g = g.sort_values("date").copy()
        g["ret"] = np.log(g["close"]).diff()

        ex = face_impulses[face_impulses["ticker"]==tkr].sort_values("date").copy()
        ex_cols = [c for c in ex.columns if c.startswith("FACE_IMP_")]

        # Align
        df = g.merge(ex[["date"] + ex_cols], on="date", how="left").fillna(0.0)

        train = df[df["date"] <= cutoff].dropna(subset=["ret"])
        test  = df[df["date"] >  cutoff]

        if len(train) < 60:
            continue

        # Fit ARIMA
        try:
            arima = ARIMA(train["ret"], order=(1,0,0)).fit()
            sigma_ar = arima.resid.std() + 1e-8
        except Exception:
            arima = None

        # Fit ARIMAX if exogs exist
        arimax = None
        if ex_cols:
            try:
                arimax = ARIMA(train["ret"], order=(1,0,0), exog=train[ex_cols]).fit()
                sigma_ax = arimax.resid.std() + 1e-8
            except Exception:
                arimax = None

        # Evaluate only on *event dates* in the test period
        evs_t = events_df[(events_df["ticker"]==tkr) & (events_df["call_date"]>cutoff)]
        for _, ev in evs_t.iterrows():
            d = ev["call_date"]
            if d not in set(test["date"]):  # next trading day might be missing
                continue
            # ARIMA forecast
            p_ar = np.nan; p_ax = np.nan
            try:
                mu_ar = arima.get_forecast(steps=(len(test[test["date"]<=d])+1)).predicted_mean.values[-1]
                p_ar = float(1 - norm.cdf(0, loc=mu_ar, scale=sigma_ar))
            except Exception:
                pass

            # ARIMAX forecast (uses exog at t+1 if available; for next-day forecast we pass the exog row at next day)
            try:
                steps = len(test[test["date"]<=d])+1
                ex_future = df.loc[df["date"] > cutoff, ex_cols].iloc[:steps]
                mu_ax = arimax.get_forecast(steps=steps, exog=ex_future).predicted_mean.values[-1]
                p_ax = float(1 - norm.cdf(0, loc=mu_ax, scale=sigma_ax))
            except Exception:
                pass

            results.append({
                "event_id": ev["event_id"],
                "ticker": tkr,
                "call_date": d,
                "p_arima": p_ar,
                "p_arimax": p_ax
            })
    return pd.DataFrame(results)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, required=True, help="Directory containing input CSVs")
    ap.add_argument("--test-split", type=float, default=0.2, help="Fraction for test (by time)")
    ap.add_argument("--embargo-days", type=int, default=3, help="Embargo gap around split (event-level tracks)")
    ap.add_argument("--arimax-decay", type=str, default="1.0,0.5,0.25", help="Comma-separated decay weights for face impulses (h=0..H)")
    args = ap.parse_args()

    data_dir = args.data_dir
    out_dir = os.path.join(data_dir, "results")
    ensure_dir(out_dir)

    # Load required files
    faces = pd.read_csv(os.path.join(data_dir, "faces_timeseries.csv"))
    events = pd.read_csv(os.path.join(data_dir, "events.csv"))
    prices = pd.read_csv(os.path.join(data_dir, "prices.csv"))

    # Parse dates
    for c in ["ts"]:
        if c in faces.columns:
            faces[c] = pd.to_datetime(faces[c], errors="coerce")
    for c in ["call_date"]:
        if c in events.columns:
            events[c] = pd.to_datetime(events[c], errors="coerce").dt.normalize()
    for c in ["date"]:
        if c in prices.columns:
            prices[c] = pd.to_datetime(prices[c], errors="coerce").dt.normalize()

    # Optional controls & VLM
    controls_event = None
    ce_path = os.path.join(data_dir, "controls_event.csv")
    if os.path.exists(ce_path):
        controls_event = pd.read_csv(ce_path)

    vlm = None
    vlm_path = os.path.join(data_dir, "vlm_event_embeddings.csv")
    if os.path.exists(vlm_path):
        vlm = pd.read_csv(vlm_path)

    # Build face features per event
    face_feats = build_face_event_features(faces)

    # Add price controls and labels
    events = add_price_controls(events, prices)
    events = build_labels_nextday(events, prices)

    # ---- Track A: Event-level models ----
    res_event = run_event_level_models(
        events=events[["event_id","ticker","call_date","direction","ret_next","mom5","mom20","vol20"]].copy(),
        face_feats=face_feats,
        controls_event=controls_event,
        vlm=vlm,
        test_frac=args.test_split,
        embargo_days=args.embargo_days
    )

    # Save predictions
    preds_ev_path = os.path.join(out_dir, "preds_event_level.csv")
    res_event["preds"].to_csv(preds_ev_path, index=False)

    # ---- Track B: Local projections ----
    # Prepare combined frame for LP
    ev_all = events.merge(face_feats, on="event_id", how="left")
    ctrl_cols = [c for c in ["mom5","mom20","vol20"] if c in ev_all.columns]
    face_cols = [c for c in ev_all.columns if c.startswith(("val_","aro_","int_"))]

    lp_df = run_local_projections(
        events_with_feats=ev_all[["event_id","ticker","call_date"] + ctrl_cols + face_cols].copy(),
        prices_df=prices,
        face_cols=face_cols,
        ctrl_cols=ctrl_cols,
        horizons=[1,2,3],
        test_frac=args.test_split,
        embargo_days=args.embargo_days
    )
    lp_path = os.path.join(out_dir, "local_projections_summary.csv")
    lp_df.to_csv(lp_path, index=False)

    # ---- Track C: ARIMA vs ARIMAX with impulses ----
    decay = parse_decay(args.arimax_decay)
    ev_face = events.merge(face_feats, on="event_id", how="left")
    exog = build_face_impulses(ev_face, prices, decay=decay)

    ar_df = run_arima_arimax(prices, events[["event_id","ticker","call_date"]], exog, test_frac=args.test_split)
    if ar_df is None or ar_df.empty:
        print("[WARN] ARIMA/ARIMAX produced no forecasts; using empty placeholder frame.")
        ar_df = events[["event_id","ticker","call_date"]].copy()
        ar_df["p_arima"] = np.nan
        ar_df["p_arimax"] = np.nan

    # Merge with ground truth for evaluation
    y_true = events[["event_id","direction"]]
    ar_df = ar_df.merge(y_true, on="event_id", how="left")
    ar_df["d_arima"]  = sign_prob_to_dir(ar_df["p_arima"])
    ar_df["d_arimax"] = sign_prob_to_dir(ar_df["p_arimax"])
    # Metrics (drop NaNs)
    eval_ar = ar_df.dropna(subset=["direction"])
    def _met(col):
        if col not in eval_ar.columns:
            return {}
        p = eval_ar[col].values
        if col.startswith("p_"):
            mask = np.isfinite(p)
            if mask.sum() == 0:
                return {"MDA": np.nan, "AUC": np.nan, "Brier": np.nan}
            y = eval_ar["direction"].values[mask]
            p_masked = p[mask]
            d = sign_prob_to_dir(p_masked)
            return {
                "MDA": mda(y, d),
                "AUC": safe_auc(y, p_masked),
                "Brier": brier(y, p_masked)
            }
        return {}
    met_arima  = _met("p_arima")
    met_arimax = _met("p_arimax")

    # Save ARIMA/ARIMAX predictions
    preds_ar_path = os.path.join(out_dir, "preds_arima_arimax.csv")
    ar_df.to_csv(preds_ar_path, index=False)

    # ---- Summary report ----
    summary_lines = []
    summary_lines.append("=== Event-level (next-day direction) ===")
    mc = res_event["metrics"]["CTRL"]
    mf = res_event["metrics"]["CTRL+FACE"]
    summary_lines.append(f"CTRL only     -> MDA: {mc['MDA']:.3f} | AUC: {mc['AUC']:.3f} | Brier: {mc['Brier']:.3f}")
    summary_lines.append(f"CTRL + FACE   -> MDA: {mf['MDA']:.3f} | AUC: {mf['AUC']:.3f} | Brier: {mf['Brier']:.3f}")
    summary_lines.append(f"ΔMDA (CF-CTRL): {mf['MDA'] - mc['MDA']:+.3f}")
    summary_lines.append(f"McNemar p (dir, CTRL vs CTRL+FACE): {res_event['mcnemar_p']}")
    if res_event["metrics"].get("CTRL+FACE+VLM"):
        mv = res_event["metrics"]["CTRL+FACE+VLM"]
        summary_lines.append(f"CTRL + FACE + VLM -> MDA: {mv['MDA']:.3f} | AUC: {mv['AUC']:.3f} | Brier: {mv['Brier']:.3f}")

    summary_lines.append("")
    summary_lines.append("=== Local Projections (coeffs/p-values for faces; OOS dir acc) ===")
    if len(lp_df):
        lp_agg = lp_df.pivot_table(index=["split","h_days"], values=["oos_dir_acc","faces_coef_L2","faces_min_p"], aggfunc="mean")
        summary_lines.append(lp_agg.to_string())
    else:
        summary_lines.append("Insufficient data for LP.")

    summary_lines.append("")
    summary_lines.append("=== ARIMA vs ARIMAX (impulses) ===")
    summary_lines.append(f"ARIMA  -> MDA: {met_arima.get('MDA', float('nan')):.3f} | AUC: {met_arima.get('AUC', float('nan')):.3f} | Brier: {met_arima.get('Brier', float('nan')):.3f}")
    summary_lines.append(f"ARIMAX -> MDA: {met_arimax.get('MDA', float('nan')):.3f} | AUC: {met_arimax.get('AUC', float('nan')):.3f} | Brier: {met_arimax.get('Brier', float('nan')):.3f}")

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    print("\n".join(summary_lines))
    print(f"\nSaved: {summary_path}")
    print(f"Per-event preds (event-level): {preds_ev_path}")
    print(f"Per-event preds (ARIMA/ARIMAX): {preds_ar_path}")
    print(f"Local projections summary: {lp_path}")


if __name__ == "__main__":
    main()
