import json
import codecs
import sys
import torch

from einops import rearrange, einsum
from tests.adapters import run_train_bpe
from cs336_basics.bpe import BPETokenizer
from cs336_basics.train import train


if __name__ == '__main__':

    configure_obj = json.loads(codecs.open(sys.argv[1], 'r', 'utf-8').read())
    if configure_obj["Type"] == "Tokenize":
        bpe_tokenizer =   BPETokenizer(configure_obj["VocabSize"], configure_obj["SpecialTokens"])
        bpe_tokenizer.train(configure_obj["TrainData"], configure_obj["WorkingDir"] + "/vocab.json", configure_obj["WorkingDir"] + "merge.json")
    elif configure_obj["Type"] == "Train":
        train(configure_obj)