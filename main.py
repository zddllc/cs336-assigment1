from tests.adapters import run_train_bpe
import json
import codecs

if __name__ == '__main__':

    # d = json.loads(codecs.open("./tests/fixtures/gpt2_vocab.json", 'r', encoding='utf-8').read())
    run_train_bpe("./tests/fixtures/corpus.en", vocab_size=500, special_tokens=["<|endoftext|>"])
