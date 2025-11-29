from tests.adapters import run_train_bpe
import json
import codecs
import torch

if __name__ == '__main__':

    from cs336_basics.bpe import split_by_special
    from cs336_basics.linear import LinearModule
    ll = LinearModule(10, 5, "cpu", torch.float32)
    split_by_special("Hello<|endoftext|>World<|startoftext|>fuck", ["<|endoftext|>", "<|startoftext|>"], False)