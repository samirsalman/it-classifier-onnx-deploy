from typer import run, Option
from model.lstm_model import LSTMClassifier
import torch
from pathlib import Path


def to_onnx(
    checkpoint: str = Option(..., "-m", "--model"),
    out_path: str = Option("runtime", "-o", "--out"),
):
    model = LSTMClassifier.load_from_checkpoint(checkpoint_path=checkpoint)
    input_sample = torch.randint(0, 100, (1, 128))
    out = Path(out_path)
    out.mkdir(parents=True, exist_ok=True)
    out = out / "model.onnx"
    model.to_onnx(
        out,
        input_sample,
        export_params=True,
        input_names=["sequence"],
        output_names=["output"],
        dynamic_axes={
            "sequence": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )


if __name__ == "__main__":
    run(to_onnx)
