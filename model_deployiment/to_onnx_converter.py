from typer import run, Option
from model_code.model.lstm_model import LSTMClassifier
import torch
from pathlib import Path


def to_onnx(
    checkpoint: str = Option(..., "-m", "--model"),
    out_path: str = Option(..., "-o", "--out"),
    input_shape: tuple = Option((1, 1, 128)),
):
    model = LSTMClassifier.load_from_checkpoint(checkpoint_path=checkpoint)
    filepath = Path(f"{out_path}/model.onnx")
    input_sample = torch.randn(input_shape)
    model.to_onnx(filepath, input_sample, export_params=True)


if __name__ == "__main__":
    run(to_onnx)
