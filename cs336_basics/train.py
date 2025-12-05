
import numpy as np
import os
import sys

from cs336_basics.bpe import BPETokenizer
from cs336_basics.utils import get_batch
from cs336_basics.module import TransformerLMModule


def generate_token_file(toknizer, train_file, token_file):


    with open(train_file, "r", encoding="utf-8") as fin, \
        open(token_file, "wb") as fout:

        index = 1
        for line in fin:
            # 对每行进行 tokenize
            ids = toknizer.encode(line)
            arr = np.array(ids, dtype=np.int16)

            # 直接追加写入二进制文件（不占内存）
            arr.tofile(fout)
            if index == 1000000:
                print("First 10 lines tokenized.")
                break
            index += 1
            sys.stdout.write(f"\rTokenizing line: {index}")

def train(configure_obj):

    bpe = BPETokenizer(configure_obj["VocabSize"], configure_obj["SpecialTokens"])
    bpe.from_files(configure_obj["WorkingDir"] + "/vocab.json", configure_obj["WorkingDir"] + "merge.json", configure_obj["SpecialTokens"])

    token_bin_file = configure_obj["WorkingDir"] + "token.bin"
    if not os.path.exists(token_bin_file):
        generate_token_file(bpe, configure_obj["TrainData"], token_bin_file)

    tokens = np.memmap(token_bin_file, dtype=np.int16, mode='r')
    x_train, y_train = get_batch(tokens, configure_obj["BatchSize"], configure_obj["ContextLength"], configure_obj["Device"])

    tllm = TransformerLMModule(
        vocab_size=configure_obj["VocabSize"],
        context_length=configure_obj["ContextLength"],
        d_model=configure_obj["EmbeddingSize"],
        num_layers=configure_obj["LayerNum"],
        num_heads=configure_obj["HeadNum"],
        d_ff=configure_obj["DFF"],
        rope_theta=None,
        device=configure_obj["Device"]
    )

    tllm.to(configure_obj["Device"])
    dd = tllm.parameters()
    for name, param in tllm.named_parameters():
        print(name, param.shape)
    
    total = sum(p.numel() for p in tllm.parameters())
    print("Total parameters:", total)
    
    stop = 1


