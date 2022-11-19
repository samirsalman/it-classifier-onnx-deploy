from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchtext.vocab import Vocab
import torch
import torch.nn as nn
import torchtext


class TextDataset(Dataset):
    def __init__(
        self,
        dataset: pd.DataFrame,
        vocab: Vocab,
        text_column: str = "Text",
        target_column: str = "target",
        max_length: int = 128,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.text_column = text_column
        self.target_column = target_column
        self.texts = dataset[text_column].to_numpy()
        self.targets = dataset[target_column].to_numpy()
        self.vocab = vocab
        self.vocab.set_default_index(0)
        self.max_length = max_length

    def __getitem__(self, index):
        tokenization = self.vocab(tokens=self.texts[index].split())
        tokenization = torchtext.transforms.Truncate(max_seq_len=self.max_length)(
            tokenization
        )
        tokenization = torch.tensor(tokenization)
        tokenization = torchtext.transforms.PadTransform(
            max_length=self.max_length, pad_value=self.vocab["<pad>"]
        )(tokenization)
        return {
            "text": self.texts[index],
            "target": self.targets[index],
            "tokenization": tokenization,
        }

    def __len__(self):
        return len(self.texts)
