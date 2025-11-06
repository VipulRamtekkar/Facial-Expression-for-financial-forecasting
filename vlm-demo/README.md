# VLM Demo

Quick demo pipeline that samples sparse frames from a CEO interview, scores them with an OpenRouter vision-language model, aggregates affect features, and combines them with price data for directional predictions.

## Setup

1. Create and activate a Python environment (3.9+ recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file and populate it with your OpenRouter API key and preferred vision model. Minimum:
   ```
   OPENROUTER_API_KEY="ENTER OPENROUTER KEY HERE"
   # optional override (defaults to meta-llama/llama-3.2-11b-vision-instruct)
   VLM_MODEL_ID=openai/gpt-5-mini
   ```
4. Add input assets:
   - Place the interview video under `data/videos/`.
   - Put the matching minute bars CSV under `data/prices/` with columns `timestamp,open,high,low,close,volume`.

## Running the demo

```bash
python -m src.demo \
  --ticker DAL \
  --video data/videos/dal_ceo_clip.mp4 \
  --frames-dir data/frames/dal_clip \
  --bars data/prices/DAL_1min.csv \
  --event-time 2024-05-10T14:30:00 \
  --t-start 0 \
  --t-end 210 \
  --frame-stride 30 \
  --horizons 5 60
```

Key flags:
- `--frame-stride`: seconds between sampled frames (use 30 for quick tests; lower when ready for higher fidelity).
- `--window`: aggregation window size in seconds for VLM features.
- `--min-conf`: drop frames whose VLM confidence falls below this threshold.

If price labels are unavailable or a class imbalance prevents model fitting, the script falls back to printing the direction implied by the observed label.

## Code overview

- `src/demo.py`: end-to-end orchestrator that extracts frames, scores them with the VLM, builds features, joins price signals, and trains/log-evaluates a directional classifier for each requested horizon.
- `src/extract_frames.py`: OpenCV-based sampler; can be called directly with `python -m src.extract_frames --video ... --out-dir ...` to inspect intermediate PNGs.
- `src/openrouter_vlm.py`: wraps the OpenRouter chat completions API; expects `OPENROUTER_API_KEY` and optional `VLM_MODEL_ID` in `.env`, returning structured affect scores per frame.
- `src/features_vlm.py`: aggregates per-frame VLM outputs into windowed statistics (means, std, lags) after filtering by minimum confidence.
- `src/features_price.py`: loads minute-bar CSVs, aligns them to the event timestamp, fills small gaps with time interpolation, and produces price-derived features plus future-return labels.
- `src/train_eval.py`: light sklearn pipeline (scaler + logistic regression) that trains/validates directional models, falling back to fitting on full data when samples are scarce.
- `src.fetch_prices.py`: CLI utility described below that infers the publish time from `ffprobe` metadata (or a manual override) and saves Yahoo Finance OHLCV bars.
- `src/align_label.py`: helper for experiments that need to attach shared labels to every scored frame row.

## Monitoring & GUI

- All CLI entry points emit structured logs (timestamp, level, module) so you can trace extraction, scoring, feature building, and model training progress when running `python -m src.demo ...`.
- Launch the interactive dashboard with:
  ```bash
  streamlit run src/app.py
  ```
  The app guides you through the full pipeline, visualizes frame-level VLM scores, plots aggregated features, surfaces derived price signals, and lists per-horizon directional predictions.
- Prefer a notebook walkthrough? Open `src/vlm_frame_walkthrough.ipynb` for a step-by-step inspection of frames, VLM outputs, aggregated features, and resulting labels.

## Pulling price data from video metadata

Use the helper to infer a video's publish time (via `ffprobe`) and fetch surrounding minute bars with Yahoo Finance:

```bash
python -m src.fetch_prices \
  --video data/videos/dal_ceo_clip.mp4 \
  --ticker DAL \
  --output-dir data/prices \
  --window-before 240 \
  --window-after 240
```

If the video file lacks a `creation_time` tag, pass `--event-time 2024-05-10T14:30:00`. The script saves `"{TICKER}_{interval}.csv"` under `data/prices/` and automatically falls back to 5-minute bars when 1-minute data is unavailable.
