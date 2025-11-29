from tests.adapters import run_train_bpe
import json
import codecs
import torch
from einops import rearrange, einsum

if __name__ == '__main__':

    from cs336_basics.bpe import split_by_special
    mask = torch.tril(torch.ones((8, 8), dtype=torch.bool, device="cpu"))
    dd = rearrange(mask, "i (m n) -> i n m", m=4)
    print(dd)

