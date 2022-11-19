import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from data.loaders.text_dataset import TextDataset
from model.lstm_model import LSTMClassifier
from torch.utils.data import DataLoader
import pandas as pd
import pytorch_lightning.callbacks as callbacks
from typer import Option, run


# wandb run: https://wandb.ai/samirsalman/Lang-Classifier/runs/1yolc4oy?workspace=user-samirsalman
pl.seed_everything(12)


def train(
    batch_size: int = Option(32, "-b", "--batch-size"),
    train_df: str = Option("model_code/data/datasets/train.csv", "-t", "--train"),
    val_df: str = Option("model_code/data/datasets/val.csv", "-v", "--val"),
    test_df: str = Option("model_code/data/datasets/test.csv", "-T", "--test"),
    vocab: str = Option("model_code/data/vocabulary/vocab.pth", "-v", "--vocab"),
    dropout: float = Option(0.2, "-d", "--dropout"),
    embedding_size: int = Option(256, "-s", "--emb-size"),
    lr: float = Option(2e-2, "-l", "--lr"),
    max_epochs: int = Option(5, "-e", "--epochs"),
    gpu: bool = Option(True, "-g", "--gpu"),
):
    train = pd.read_csv(train_df)
    val = pd.read_csv(val_df)
    test = pd.read_csv(test_df)
    vocab = torch.load(vocab)
    train_dataset = TextDataset(dataset=train, vocab=vocab, max_length=32)
    val_dataset = TextDataset(dataset=val, vocab=vocab, max_length=32)
    test_dataset = TextDataset(dataset=test, vocab=vocab, max_length=32)

    trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(dataset=val_dataset, batch_size=batch_size)
    testloader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    model = LSTMClassifier(
        vocab=vocab,
        dropout=dropout,
        embedding_size=embedding_size,
        lr=lr,
    )

    trainer = pl.Trainer(
        deterministic=True,
        max_epochs=max_epochs,
        callbacks=[
            callbacks.ModelCheckpoint(
                dirpath="model", filename="{epoch}-{val_loss:.2f}"
            ),
            callbacks.EarlyStopping(monitor="val_f1_score", patience=2, mode="max"),
        ],
        logger=WandbLogger(project="Lang-Classifier"),
        precision=16,
        accelerator="gpu" if gpu else None,
    )
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=valloader)


if __name__ == "__main__":
    run(train)
