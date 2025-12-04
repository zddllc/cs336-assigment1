
import numpy as np
import os

from cs336_basics.bpe import BPETokenizer
from cs336_basics.utils import get_batch

def generate_token_file(toknizer, train_file, token_file):


    with open(train_file, "r", encoding="utf-8") as fin, \
        open(token_file, "wb") as fout:

        for line in fin:
            # 对每行进行 tokenize
            ids = toknizer.encode(line)
            arr = np.array(ids)

            # 直接追加写入二进制文件（不占内存）
            arr.tofile(fout)

def train(configure_obj):

    bpe = BPETokenizer(configure_obj["VocabSize"], configure_obj["SpecialTokens"])
    bpe.from_files(configure_obj["WorkingDir"] + "/vocab.json", configure_obj["WorkingDir"] + "merge.json", configure_obj["SpecialTokens"])

    token_bin_file = configure_obj["WorkingDir"] + "token.bin"
    if not os.path.exists(token_bin_file):
        generate_token_file(bpe, configure_obj["TrainData"], token_bin_file)

    tokens = np.memmap(token_bin_file, dtype=np.int32, mode='r')
    x_train, y_train = get_batch(tokens, configure_obj["BatchSize"], configure_obj["ContextLength"], configure_obj["Device"])