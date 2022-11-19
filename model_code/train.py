import torch
import pytorch_lightning as pl
from data.loaders.text_dataset import TextDataset
from model.lstm_model import LSTMClassifier
from torch.utils.data import DataLoader
import pandas as pd
import pytorch_lightning.callbacks as callbacks

pl.seed_everything(12)

train = pd.read_csv("model_code/data/datasets/train.csv")
val = pd.read_csv("model_code/data/datasets/val.csv")
test = pd.read_csv("model_code/data/datasets/test.csv")
vocab = torch.load("model_code/data/vocabulary/vocab.pth")
train_dataset = TextDataset(dataset=train, vocab=vocab)
val_dataset = TextDataset(dataset=val, vocab=vocab)
test_dataset = TextDataset(dataset=test, vocab=vocab)

trainloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
valloader = DataLoader(dataset=val_dataset, batch_size=32)
testloader = DataLoader(dataset=test_dataset, batch_size=32)

model = LSTMClassifier(
    vocab=torch.load("model_code/data/vocabulary/vocab.pth"),
    dropout=0.2,
    embedding_size=256,
    lr=2e-2,
)

trainer = pl.Trainer(
    deterministic=True,
    max_epochs=3,
    callbacks=[
        callbacks.ModelCheckpoint(dirpath="model", filename="{epoch}-{val_loss:.2f}"),
        callbacks.EarlyStopping(monitor="val_f1_score", patience=2, mode="max"),
    ],
    precision=16,
    accelerator="gpu",
)
trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=valloader)
