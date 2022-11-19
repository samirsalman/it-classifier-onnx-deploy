import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from torchtext.vocab import Vocab
from torchmetrics import Accuracy
from torchmetrics.classification.f_beta import BinaryF1Score
import torchtext


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
        max_len=str=128,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.vocab = vocab
        self.lr = lr
        self.dropout = nn.Dropout(dropout)
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.criterion = nn.CrossEntropyLoss()
        self.max_len = max_len

        self.embedding = nn.Embedding(
            num_embeddings=len(vocab) + 1,
            embedding_dim=self.embedding_size,
            padding_idx=1,
        )

        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Linear(self.hidden_size, 2)
        self.accuracy = Accuracy()
        self.f1_score = BinaryF1Score()

    def init_state(self, x):
        return (
            torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device),
            torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device),
        )

    def forward(self, x):
        h0 = self.init_state(x=x)
        output, state = self.lstm(x, h0)
        output = self.dropout(output)
        output = output[:, -1, :]
        output = self.ffn(output)
        return output, state

    def training_step(self, batch, batch_idx):
        text = batch["text"]
        target = batch["target"]
        tokenization = batch["tokenization"]
        emb = self.embedding(tokenization)
        logits, _ = self(emb)

        loss_value = self.criterion(logits, target.to(torch.long))
        preds = torch.argmax(logits, dim=1)
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
        logits, _ = self(emb)

        loss_value = self.criterion(logits, target.to(torch.long))
        preds = torch.argmax(logits, dim=1)
        accuracy = self.accuracy(preds, target)
        f_score = self.f1_score(preds, target)

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
        logits, _ = self(emb)
        loss_value = self.criterion(logits, target.to(torch.long))
        preds = torch.argmax(logits, dim=1)

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
