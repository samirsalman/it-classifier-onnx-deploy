import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from torchtext.vocab import Vocab
from torchmetrics import Accuracy
from torchmetrics.classification.f_beta import BinaryF1Score


class LSTMClassifier(pl.LightningModule):
    def __init__(
        self,
        vocab: Vocab,
        lr: float = 2e-3,
        dropout: float = 0.1,
        embedding_size: int = 256,
        hidden_size: int = 128,
        lstm_size: int = 256,
        num_layers: int = 5,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.vocab = vocab
        self.lr = lr
        self.dropout = dropout
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.criterion = nn.BCEWithLogitsLoss()

        self.embedding = nn.Embedding(
            num_embeddings=len(vocab) + 1,
            embedding_dim=self.embedding_size,
            padding_idx=1,
        )

        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=dropout,
        )
        self.relu = nn.ReLU()
        self.ffn = nn.Linear(self.lstm_size, 1)
        self.accuracy = Accuracy()
        self.f1_score = BinaryF1Score()

    def init_state(self, x):
        return (
            torch.zeros(self.num_layers, x.size(1), self.lstm_size).to(self.device),
            torch.zeros(self.num_layers, x.size(1), self.lstm_size).to(self.device),
        )

    def forward(self, x, h0):
        output, state = self.lstm(x, h0)
        output = output[:, -1, :]
        output = self.relu(output)
        logits = self.ffn(output)
        return logits, state

    def training_step(self, batch, batch_idx):
        text = batch["text"]
        target = batch["target"]
        tokenization = batch["tokenization"]
        emb = self.embedding(tokenization)
        state = self.init_state(emb)
        logits, _ = self(emb, state)
        logits = logits.view(-1)
        loss_value = self.criterion(logits, target.to(torch.float32))
        preds = torch.round(logits)
        accuracy = self.accuracy(preds, target)
        f_score = self.f1_score(preds, target)
        self.log_dict(
            {"loss": loss_value, "train_accuracy": accuracy, "train_f1_score": f_score},
            prog_bar=True,
            on_step=True,
        )
        return loss_value

    def validation_step(self, batch, batch_idx):
        text = batch["text"]
        target = batch["target"]
        tokenization = batch["tokenization"]
        emb = self.embedding(tokenization)
        state = self.init_state(emb)
        logits, _ = self(emb, state)
        logits = logits.view(-1)
        loss_value = self.criterion(logits, target.to(torch.float32))
        preds = torch.round(logits)
        accuracy = self.accuracy(preds, target)
        f_score = self.f1_score(preds, target)

        print(
            "text", text[0], "target", target[0].item(), "prediction", preds[0].item()
        )
        self.log_dict(
            {"val_loss": loss_value, "val_accuracy": accuracy, "val_f1_score": f_score},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return loss_value

    def test_step(self, batch, batch_idx):
        text = batch["text"]
        target = batch["target"]
        tokenization = batch["tokenization"]
        emb = self.embedding(tokenization)
        state = self.init_state(emb)
        logits, _ = self(emb, state)
        logits = logits.view(-1)
        loss_value = self.criterion(logits, target.to(torch.float32))
        preds = torch.round(logits)
        accuracy = self.accuracy(preds, target)
        f_score = self.f1_score(preds, target)

        self.log_dict(
            {
                "test_loss": loss_value,
                "test_accuracy": accuracy,
                "test_f1_score": f_score,
            },
            prog_bar=True,
        )
        return loss_value

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
