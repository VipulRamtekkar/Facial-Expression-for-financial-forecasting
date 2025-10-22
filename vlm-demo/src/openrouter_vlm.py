import base64
import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_ID = os.getenv("VLM_MODEL_ID", "meta-llama/llama-3.2-11b-vision-instruct")
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = (
    "You analyze close-up images of a person during a live interview. "
    "Return STRICT JSON with keys: valence (-1..1), arousal (0..1), "
    "intensity (0..1), confidence (0..1), notes (short). "
    "No markdown. No extra text. If uncertain, keep confidence < 0.5."
)

USER_TEXT = (
    "This is a single video frame from a CEO interview. "
    "Estimate affect signals that are temporally local to this frame."
)


def _encode_png(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def score_frame(frame_path, system_prompt=SYSTEM_PROMPT, user_text=USER_TEXT):
    if not OPENROUTER_KEY:
        raise RuntimeError("OPENROUTER_API_KEY missing. Populate .env before running.")

    image_b64 = _encode_png(frame_path)
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            },
        ],
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "HTTP-Referer": "https://localhost/cli",
        "X-Title": "vlm-demo",
    }
    response = requests.post(ENDPOINT, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)
