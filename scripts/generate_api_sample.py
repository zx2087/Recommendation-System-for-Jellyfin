"""
Generate contracts/input_sample.json (API request format) from the training sample.

The training sample (recommender_input.sample.json) stores single user-movie pairs
used for model training.  The serving API expects a different format: one user
embedding + a list of candidate movies.  This script bridges the gap.

Run from repo root:
    python scripts/generate_api_sample.py
"""

import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_training_sample() -> dict:
    path = ROOT / "contracts" / "recommender_input.sample.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Training sample is a list; take the first entry
    return data[0] if isinstance(data, list) else data


def make_api_input(sample: dict, n_candidates: int = 20) -> dict:
    """
    Build a RecommendRequest payload using real embeddings from the training sample.

    - candidate_movies[0] uses the REAL movie embedding from the sample
    - candidates[1..n-1] are small L2-norm-preserving perturbations of the real
      embedding, simulating semantically similar movies in the same embedding space
    """
    rng = np.random.default_rng(42)

    user_emb = np.array(sample["user_embedding"], dtype=np.float32)
    real_movie_emb = np.array(sample["movie_embedding"], dtype=np.float32)
    real_movie_norm = float(np.linalg.norm(real_movie_emb))

    candidate_movies = [
        {
            "movie_id": f"movie_{sample['movie_id']}",   # keep original id as string
            "movie_embedding": real_movie_emb.tolist(),
        }
    ]

    for i in range(1, n_candidates):
        # Perturb real embedding slightly (noise std = 2 % of L2 norm)
        noise = rng.normal(0, 0.02, size=real_movie_emb.shape).astype(np.float32)
        synth = real_movie_emb + noise
        # Rescale to original L2 norm so the magnitude stays representative
        synth = synth / (float(np.linalg.norm(synth)) + 1e-8) * real_movie_norm
        candidate_movies.append({
            "movie_id": f"movie_synth_{i:04d}",
            "movie_embedding": synth.tolist(),
        })

    return {
        "request_id": "req_160710_001",
        "user_id": f"user_{sample['user_id']}",   # int → string
        "timestamp": "2025-01-15T10:30:00Z",
        "request_k": 10,
        "user_embedding": user_emb.tolist(),
        "candidate_movies": candidate_movies,
        "client_context": {
            "surface": "homepage",
            "device_type": "web",
            "app_version": "0.1.0",
        },
    }


def main() -> None:
    sample = load_training_sample()
    api_input = make_api_input(sample, n_candidates=20)

    out_path = ROOT / "contracts" / "input_sample.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(api_input, f, indent=2)

    print(f"Written  : {out_path}")
    print(f"user_id  : {api_input['user_id']}")
    print(f"user_emb : dim={len(api_input['user_embedding'])}")
    print(f"candidates: {len(api_input['candidate_movies'])}"
          f" (1 real + {len(api_input['candidate_movies'])-1} synthetic)")


if __name__ == "__main__":
    main()
