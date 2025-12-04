import codecs
import json
import os
import multiprocessing
import regex
import re

from collections import defaultdict
from typing import BinaryIO

PAT = r"""'(?i:[sdmt]|ll|ve|re)|
     [^\r\n\p{L}\p{N}]?+[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|
     [^\r\n\p{L}\p{N}]?+[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|
     \p{N}{1,3}|
     ?[^\s\p{L}\p{N}]++[\r\n]*|
     \s*[\r\n]|
     \s+(?!\S)|
     \s+"""
PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

_unused_pat = regex.compile(PAT)


def bytes_to_unicode():

    bytes_list = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    coverted_bytes = bytes_list[:]

    n = 0
    for b in range(256):
        if b not in bytes_list:
            bytes_list.append(b)
            coverted_bytes.append(256 + n)
            n += 1

    printble = [chr(n) for n in coverted_bytes]
    return dict(zip(bytes_list, printble))

printable_dict = bytes_to_unicode()
revert_dict = {v: k for k, v in printable_dict.items()}


def split_by_special(text, special_tokens, drop_special=True):

    if not special_tokens:
        return [text]

    # Sort by descending length to prioritize longer tokens (e.g., "<|endoftext|><|endoftext|>" before "<|endoftext|>")
    special_tokens = sorted(special_tokens, key=len, reverse=True)

    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    if not drop_special: 
        pattern = f"({pattern})"

    pattern = re.compile(pattern)
    chunks = pattern.split(text)
    return [c for c in chunks if c]


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenize_chunk(input_path, chunk, special_tokens) -> dict[str, int]:

    """
    Example pre-tokenization function that counts word frequencies in a chunk.
    """
    from collections import defaultdict

    start = chunk[0]
    end = chunk[1]
    ret = defaultdict(int)

    with open(input_path, "rb") as f:
        f.seek(start)

        chunk_text = f.read(end - start).decode("utf-8", errors="ignore")
        chunk_text = chunk_text.replace("\r\n", "\n").replace("\r", "")
        chunks = split_by_special(chunk_text, special_tokens)

        for chunk_text in chunks:
            words = regex.finditer(_unused_pat, chunk_text)
            for one_token in words:
                ret[one_token.group()] += 1

        return ret

    return {}

class PreToken(object):

    def __init__(self, text, count):

        self.text = text
        self.count = count
        self.utf8_str = text.encode("utf-8")
        self.utf8_str_list = []
        for one_byte in self.utf8_str:
            self.utf8_str_list.append(bytes([one_byte]))

        pass


class BPETokenizer(object):

    def __init__(self, vocab_size, special_tokens):

        self.input_path = None
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens if special_tokens is not None else []

        self.vocab_path = None
        self.merge_path = None

        self.vocab = {}
        self.vocab_byte2index = {}

        self.merge = []
        self.merge_byte2index = {}

    def train(self, input_path, vocab_path, merge_path):

        self.input_path = input_path
        self.vocab_path = vocab_path
        self.merge_path = merge_path

        merges = []
        merges_out = []

        vocab = {}
        vocab_out = {}

        vocab_index = 0
        for one_special_token in self.special_tokens:
            vocab[vocab_index] = one_special_token.encode("utf-8", errors="ignore")
            vocab_out[vocab_index] = one_special_token
            vocab_index += 1

        for assic_index in range(256):
            vocab[vocab_index] = bytes([assic_index])
            vocab_out[vocab_index] = printable_dict[assic_index]
            vocab_index += 1

        num_processes = 16
        with open(self.input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.

            chunks = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                chunks.append((start, end))
                # f.seek(start)
                # chunk = f.read(end - start).decode("utf-8", errors="ignore")

            final_dict = defaultdict(int)

            pool = multiprocessing.Pool(processes=num_processes)
            all_results = []
            for c in chunks:
                args = (self.input_path, c, self.special_tokens)
                res = pool.apply_async(pre_tokenize_chunk, args=args, )
                all_results.append(res)

            for one_result in all_results:
                output = one_result.get()
                for k, v in output.items():
                    final_dict[k] += v

            pool.close()
            pool.join()

        # import regex
        # text = codecs.open(self.input_path, "r", encoding="utf-8", errors="ignore").read()
        # chunks = split_by_special(text, self.special_tokens)
        # PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # _unused_pat = regex.compile(PAT)

        # final_dict = defaultdict(int)
        # for text in chunks:
        #     for m in _unused_pat.finditer(text):
        #         w = m.group(0)
        #         wb = tuple(bytes([b]) for b in w.encode("utf-8"))
        #         if len(wb) >= 2:
        #             final_dict[w] += 1


        pair_count, pair_index = self.init_pair_count_and_pair_index(final_dict)

        while len(vocab) < self.vocab_size:
            sorted_pairs = sorted(pair_count.items(), key=lambda x: (x[1], x[0]), reverse=True)
            select_top_pair = sorted_pairs[0]
            print(f"Current vocab size: {vocab_index} \r")
            self.update_pair_count_and_pair_index(pair_count, pair_index, select_top_pair, vocab, vocab_out, vocab_index, merges, merges_out)
            vocab_index += 1
        
        f = codecs.open(self.vocab_path, "w", encoding="utf-8")
        f.write(json.dumps(vocab_out, ensure_ascii=False, indent=4))
        f.close()

        f = codecs.open(self.merge_path, "w", encoding="utf-8")
        f.write(json.dumps(merges_out, ensure_ascii=False, indent=4))
        f.close()

        test_vocab, test_vocab_byte2index = self.convert_printable_vocab_to_memory(vocab_out)
        test_merge, test_merge_byte2index = self.convert_printable_merge_to_memory(merges_out)

        assert test_vocab == vocab
        assert test_merge == merges

        return vocab, merges


    def init_pair_count_and_pair_index(self, final_dict):

        pair_counts = defaultdict(int)
        pair_index = defaultdict(list)
        for tokenized_text, count in final_dict.items():
            pre_token = PreToken(tokenized_text, count)
            for index in range(len(pre_token.utf8_str_list) - 1):
                pair = (pre_token.utf8_str_list[index], pre_token.utf8_str_list[index + 1])
                pair_counts[pair] += count
                pair_index[pair].append(pre_token)

        return pair_counts, pair_index


    def update_pair_count_and_pair_index(self, pair_count, pair_index, select_top_pair, vocab, vocab_out, vocab_size, merges, merges_out):
            
        top_pair = select_top_pair[0]

        merge_byte = top_pair[0] + top_pair[1]

        merge_str = ''.join(printable_dict[byte] if byte < 256 else chr(byte) for byte in top_pair[0]) + " " + ''.join(printable_dict[byte] if byte < 256 else chr(byte) for byte in top_pair[1])
        vocab_str = ''.join(printable_dict[byte] if byte < 256 else chr(byte) for byte in merge_byte)
        if merge_str not in merges_out:
            merges.append((top_pair[0], top_pair[1]))
            merges_out.append(merge_str)

            vocab[vocab_size] = merge_byte
            vocab_out[vocab_size] = vocab_str
        
        new_token = merge_byte
        new_pairs = []
        old_pairs = set()

        for pre_token_index, cur_pre_token in enumerate(pair_index[top_pair]):

            while True:

                start_index = None
                for pos_index in range(len(cur_pre_token.utf8_str_list) - 1):
                    if (cur_pre_token.utf8_str_list[pos_index], cur_pre_token.utf8_str_list[pos_index + 1]) == top_pair:
                        start_index = pos_index
                        break

                if start_index is None:
                    break
                else:
                    count = cur_pre_token.count
                    
                    if start_index == 0:
                        if len(cur_pre_token.utf8_str_list) >= 3:
                            old_pair = (cur_pre_token.utf8_str_list[1], cur_pre_token.utf8_str_list[2])
                            pair_count[old_pair] -= count
                            old_pairs.add(old_pair)

                        cur_pre_token.utf8_str_list = tuple([new_token] + list(cur_pre_token.utf8_str_list[2:]))
                        if len(cur_pre_token.utf8_str_list) >= 2:
                            new_pair = (cur_pre_token.utf8_str_list[0], cur_pre_token.utf8_str_list[1])
                            pair_count[new_pair] += count
                            pair_index[new_pair].append(cur_pre_token)
                            new_pairs.append(new_pair)

                    elif start_index == len(cur_pre_token.utf8_str_list) - 2:
                        if len(cur_pre_token.utf8_str_list) >= 3:
                            old_pair = (cur_pre_token.utf8_str_list[-3], cur_pre_token.utf8_str_list[-2])
                            pair_count[old_pair] -= count
                            old_pairs.add(old_pair)

                        cur_pre_token.utf8_str_list = tuple(list(cur_pre_token.utf8_str_list[: start_index]) + [new_token])
                        if len(cur_pre_token.utf8_str_list) >= 2:
                            new_pair = (cur_pre_token.utf8_str_list[-2], cur_pre_token.utf8_str_list[-1])
                            pair_count[new_pair] += count
                            pair_index[new_pair].append(cur_pre_token)
                            new_pairs.append(new_pair)
                    else:
                        old_pair1 = (cur_pre_token.utf8_str_list[start_index - 1], cur_pre_token.utf8_str_list[start_index])
                        pair_count[old_pair1] -= count
                        old_pairs.add(old_pair1)

                        old_pair2 = (cur_pre_token.utf8_str_list[start_index + 1], cur_pre_token.utf8_str_list[start_index + 2])
                        pair_count[old_pair2] -= count
                        old_pairs.add(old_pair2)

                        cur_pre_token.utf8_str_list = tuple(list(cur_pre_token.utf8_str_list[:start_index]) + [new_token] + list(cur_pre_token.utf8_str_list[start_index + 2 :]))
                        
                        new_pair1 = (cur_pre_token.utf8_str_list[start_index - 1], new_token)
                        pair_count[new_pair1] += count
                        pair_index[new_pair1].append(cur_pre_token)
                        new_pairs.append(new_pair1)

                        new_pair2 = (new_token, cur_pre_token.utf8_str_list[start_index + 1])
                        pair_count[new_pair2] += count
                        pair_index[new_pair2].append(cur_pre_token)
                        new_pairs.append(new_pair2)

        # for one_new_pair in new_pairs:
        #     print(f'Updated pair: {one_new_pair}, count: {pair_count[one_new_pair]}')

        for one_old_pair in old_pairs:
            if pair_count[one_old_pair] <= 0:
                del pair_count[one_old_pair]
        
        if top_pair in pair_count:
            del pair_count[top_pair]

        if top_pair in pair_index:
            del pair_index[top_pair]

    def from_memory(self, vocab, merge, special_tokens):

        self.vocab = vocab
        self.merge = merge
        self.special_tokens = special_tokens if special_tokens is not None else []

        for index, one_merge in enumerate(self.merge):
            self.merge_byte2index[one_merge] = index

        for index, one_vocab in self.vocab.items():
            self.vocab_byte2index[one_vocab] = index

    def from_files(self, vocab_path, merge_path, special_tokens):
    
        self.vocab_path = vocab_path
        self.merge_path = merge_path
        self.special_tokens = special_tokens if special_tokens is not None else []

        with open(self.vocab_path, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
            self.vocab, self.vocab_byte2index = self.convert_printable_vocab_to_memory(vocab_json)

        with open(self.merge_path, "r", encoding="utf-8") as f:
            merge_json = json.load(f)
            self.merge, self.merge_byte2index = self.convert_printable_merge_to_memory(merge_json)

    def convert_printable_vocab_to_memory(self, vocab_json):

        vocab = {}
        vocab_byte2index = {}
        for k, v in vocab_json.items():
            token_byte = b"".join(bytes([int(revert_dict[c])]) if c in printable_dict.values() else c.encode("utf-8") for c in v)
            vocab[int(k)] = token_byte
            vocab_byte2index[token_byte] = int(k)

        return vocab, vocab_byte2index

    def convert_printable_merge_to_memory(self, merge_json):

        merge = []
        merge_byte2index = {}
        for one_merge_str in merge_json:
            token_strs = one_merge_str.split(" ")
            token_bytes = (b"".join(bytes([int(revert_dict[c])]) if c in printable_dict.values() else c.encode("utf-8") for c in token_strs[0]), 
                        b"".join(bytes([int(revert_dict[c])]) if c in printable_dict.values() else c.encode("utf-8") for c in token_strs[1]))
            merge.append(token_bytes)
            merge_byte2index[token_bytes] = len(merge_byte2index)

        return merge, merge_byte2index

    def encode(self, text):

        chunks = split_by_special(text, self.special_tokens, False)

        ret_id = []
        for one_chunk in chunks:
            if one_chunk in self.special_tokens:
                ret_id.append(self.vocab_byte2index[one_chunk.encode("utf-8", errors="ignore")])
            else:

                pre_tokens = regex.finditer(_unused_pat, one_chunk)
                for one_token in pre_tokens:

                    token_bytes = tuple(bytes([b]) for b in one_token.group().encode("utf-8", errors="ignore"))
                    while True:

                        all_pontential_tokens = []
                        for index in range(len(token_bytes) - 1):
                            pair = (token_bytes[index], token_bytes[index + 1])
                            if pair in self.merge_byte2index:
                                all_pontential_tokens.append([(token_bytes[index], token_bytes[index + 1]), self.merge_byte2index[pair]])

                        if len(all_pontential_tokens) == 0:
                            break

                        pair2merge = sorted(all_pontential_tokens, key=lambda x: x[1])[0][0]

                        new_token_bytes = []
                        index = 0
                        while index < len(token_bytes):
                            if index == len(token_bytes) - 1:
                                new_token_bytes.append(token_bytes[index])
                                index += 1
                            else:
                                if (token_bytes[index], token_bytes[index + 1]) == pair2merge:
                                    new_token_bytes.append(token_bytes[index] + token_bytes[index + 1])
                                    index += 2
                                else:
                                    new_token_bytes.append(token_bytes[index])
                                    index += 1

                        token_bytes = tuple(new_token_bytes)

                    token_bytes = list(token_bytes)
                    for one_token in token_bytes:
                        token_id = self.vocab_byte2index[one_token]
                        ret_id.append(token_id)

        return ret_id
    
    def encode_iterable(self, texts):

        for one_text in texts:
            arr = self.encode(one_text)
            for one_id in arr:
                yield one_id
    
    def decode(self, ids):

        ret_bytes = b""
        for one_id in ids:
            token_bytes = self.vocab[one_id]
            ret_bytes += token_bytes

        return ret_bytes.decode("utf-8", errors="replace")