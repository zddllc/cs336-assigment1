from tests.adapters import run_train_bpe
import json
import codecs

if __name__ == '__main__':

    from cs336_basics.bpe import split_by_special
    split_by_special("Hello<|endoftext|>World<|startoftext|>fuck", ["<|endoftext|>", "<|startoftext|>"], False)