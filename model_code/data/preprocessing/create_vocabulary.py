from torchtext.vocab import vocab, Vocab
import numpy as np
import torch
from typer import Option, run
import pandas as pd
from pathlib import Path


def split_tokens(texts: np.array):
    words = dict()
    for sentence in texts:
        # simple tokenization
        # lower case sentence
        for word in sentence.lower().split():
            if word not in words:  # vocabulary
                words[word] = 1
            else:
                words[word] += 1

    return words


def generate_vocab(tokens: dict, min_freq: int = 10):
    return vocab(tokens, min_freq=min_freq, specials=["<unk>", "<pad>"])


def save_vocab(vocab: Vocab, out_path: str):
    torch.save(vocab, Path(out_path / "vocab.pth"))


def create_vocabulary(
    dataset_path: str = Option(..., "-d", "--dataset"),
    text_column: str = Option("Text", "-t", "--text-column"),
    min_freq: int = Option(10, "-f", "--freq"),
    out_path: str = Option(..., "-o", "--out"),
):
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(dataset_path)
    texts = data[text_column].to_numpy()
    tokens = split_tokens(texts=texts)
    vocab = generate_vocab(tokens=tokens, min_freq=min_freq)
    save_vocab(vocab=vocab, out_path=out_path)


if __name__ == "__main__":
    run(create_vocabulary)
