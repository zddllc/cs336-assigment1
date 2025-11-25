from tests.adapters import run_train_bpe


if __name__ == '__main__':

    run_train_bpe("/home/chenli/Code/cs336-data/TinyStoriesV2-GPT4-tiny.txt", vocab_size=1000, special_tokens=["<|endoftext|>"])
