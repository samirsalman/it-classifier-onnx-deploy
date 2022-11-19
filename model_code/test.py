import torch
import pytorch_lightning as pl
from data.loaders.text_dataset import TextDataset
from model.lstm_model import LSTMClassifier
from torch.utils.data import DataLoader
import pandas as pd
from typer import Option, run


def test(
    checkpoint_path: str = Option(..., "-m", "--model"),
    test_df: str = Option("model_code/data/datasets/test.csv", "-T", "--test"),
    vocab: str = Option("model_code/data/vocabulary/vocab.pth", "-v", "--vocab"),
    gpu: bool = Option(True, "-g", "--gpu"),
):
    test = pd.read_csv(test_df)
    vocab = torch.load(vocab)
    test_dataset = TextDataset(dataset=test, vocab=vocab)
    testloader = DataLoader(dataset=test_dataset, batch_size=32)
    model = LSTMClassifier.load_from_checkpoint(checkpoint_path)
    trainer = pl.Trainer(
        deterministic=True,
        accelerator="gpu" if gpu else None,
    )
    print(trainer.test(model=model, dataloaders=testloader))


if __name__ == "__main__":
    run(test)
