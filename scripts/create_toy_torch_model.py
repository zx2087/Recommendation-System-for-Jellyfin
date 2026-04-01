from pathlib import Path
import torch

from serving.torch_model.app.model import ToyScorer, FEATURE_DIM


def main():
    torch.manual_seed(42)

    model = ToyScorer(FEATURE_DIM)
    model.eval()

    out_path = Path("serving/torch_model/models/toy_scorer.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_path)
    print(f"Saved toy torch model to: {out_path}")


if __name__ == "__main__":
    main()