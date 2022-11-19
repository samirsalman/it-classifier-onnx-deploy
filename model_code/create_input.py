import torchtext
import torch


def tokenize(vocab: torchtext.vocab.Vocab, text: str, max_length: int = 128):
    tokenization = vocab(tokens=text.split())
    tokenization = torchtext.transforms.Truncate(max_seq_len=max_length)(tokenization)
    tokenization = torch.tensor(tokenization)
    tokenization = torchtext.transforms.PadTransform(
        max_length=max_length, pad_value=vocab["<pad>"]
    )(tokenization)
    return tokenization
