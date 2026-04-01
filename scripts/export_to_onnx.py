from pathlib import Path
import torch

from serving.torch_model.app.model import ToyScorer, FEATURE_DIM


def main():
    torch.manual_seed(42)

    model = ToyScorer(FEATURE_DIM)
    state = torch.load("serving/torch_model/models/toy_scorer.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy_input = torch.randn(4, FEATURE_DIM, dtype=torch.float32)

    out_path = Path("serving/onnx/models/toy_scorer.onnx")
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