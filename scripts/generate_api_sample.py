"""
Generate contracts/recommender_input.sample.json in the unified API request format.

Usage:
    python scripts/generate_api_sample.py

Behavior:
- Prefer contracts/recommender_input.json as source if it exists.
- Otherwise reuse contracts/recommender_input.sample.json as source.
- Always write to contracts/recommender_input.sample.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PREFERRED_SOURCE_PATH = ROOT / "contracts" / "recommender_input.json"
FALLBACK_SOURCE_PATH = ROOT / "contracts" / "recommender_input.sample.json"
OUTPUT_PATH = ROOT / "contracts" / "recommender_input.sample.json"


def get_source_path() -> Path:
    if PREFERRED_SOURCE_PATH.exists():
        return PREFERRED_SOURCE_PATH
    if FALLBACK_SOURCE_PATH.exists():
        return FALLBACK_SOURCE_PATH
    raise FileNotFoundError(
        f"Could not find source sample. Tried: {PREFERRED_SOURCE_PATH}, {FALLBACK_SOURCE_PATH}"
    )


def load_source_sample() -> dict:
    source_path = get_source_path()
    with open(source_path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        if not data:
            raise ValueError(f"{source_path} is an empty list")
        return data[0]

    if isinstance(data, dict):
        return data

    raise ValueError(f"Unsupported JSON format in {source_path}: {type(data)}")


def is_new_request_format(sample: dict) -> bool:
    return "user_embedding" in sample and "candidates" in sample


def is_old_pair_format(sample: dict) -> bool:
    return (
        "user_embedding" in sample
        and "movie_embedding" in sample
        and "movie_id" in sample
    )


def make_candidates_from_pair(sample: dict, n_candidates: int = 20) -> list[dict]:
    rng = np.random.default_rng(42)

    real_movie_emb = np.asarray(sample["movie_embedding"], dtype=np.float32)
    real_movie_norm = float(np.linalg.norm(real_movie_emb)) + 1e-8

    candidates = [
        {
            "movie_id": str(sample["movie_id"]),
            "movie_embedding": real_movie_emb.tolist(),
        }
    ]

    for i in range(1, n_candidates):
        noise = rng.normal(0, 0.02, size=real_movie_emb.shape).astype(np.float32)
        synth = real_movie_emb + noise
        synth = synth / (float(np.linalg.norm(synth)) + 1e-8) * real_movie_norm

        candidates.append(
            {
                "movie_id": f"synth_{i:04d}",
                "movie_embedding": synth.tolist(),
            }
        )

    return candidates


def make_api_input(sample: dict, n_candidates: int = 20) -> dict:
    if is_new_request_format(sample):
        req = dict(sample)

        if "candidates" not in req or not isinstance(req["candidates"], list):
            raise ValueError("New-format sample must contain a list field 'candidates'")

        req.setdefault("request_id", "sample-request-001")
        req.setdefault("timestamp", "2026-04-05T20:00:00Z")
        req.setdefault("request_k", min(10, max(1, len(req["candidates"]))))
        req.setdefault(
            "client_context",
            {
                "surface": "homepage",
                "device_type": "web",
                "app_version": "0.1.0",
            },
        )
        req["user_id"] = str(req["user_id"])
        return req

    if is_old_pair_format(sample):
        user_emb = np.asarray(sample["user_embedding"], dtype=np.float32)

        return {
            "request_id": "sample-request-001",
            "user_id": str(sample["user_id"]),
            "timestamp": "2026-04-05T20:00:00Z",
            "request_k": min(10, n_candidates),
            "user_embedding": user_emb.tolist(),
            "candidates": make_candidates_from_pair(sample, n_candidates=n_candidates),
            "client_context": {
                "surface": "homepage",
                "device_type": "web",
                "app_version": "0.1.0",
            },
        }

    raise ValueError(
        "Unsupported sample format. Expected old pair format or new request format."
    )


def main() -> None:
    sample = load_source_sample()
    api_input = make_api_input(sample, n_candidates=20)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(api_input, f, indent=2)

    print(f"Written        : {OUTPUT_PATH}")
    print(f"user_id        : {api_input['user_id']}")
    print(f"user_emb_dim   : {len(api_input['user_embedding'])}")
    print(f"candidate_count: {len(api_input['candidates'])}")
    print(f"request_k      : {api_input['request_k']}")


if __name__ == "__main__":
    main()