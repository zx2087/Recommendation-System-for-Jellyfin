from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from serving.torch_model.app.model import RecommenderMLP, FEATURE_DIM


def main():
    model = RecommenderMLP(FEATURE_DIM)

    model_path = ROOT / "serving" / "torch_model" / "models" / "model_mlp_best.pt"
    print("Loading model from:", model_path)

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy_input = torch.randn(4, FEATURE_DIM, dtype=torch.float32)

    out_path = ROOT / "serving" / "onnx" / "models" / "model_mlp_best.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        out_path.as_posix(),
        input_names=["features"],
        output_names=["scores"],
        dynamic_axes={
            "features": {0: "num_candidates"},
            "scores": {0: "num_candidates"},
        },
        opset_version=17,
    )

    print(f"Exported ONNX model to: {out_path}")


if __name__ == "__main__":
    main()