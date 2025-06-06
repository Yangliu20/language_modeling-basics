import regex as re
import time
import os
from typing import BinaryIO
import multiprocessing
import json

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
    ) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

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


def pretokenize_chunk(start, end, input_path, special_tokens):
    freq = {}
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        # Run pre-tokenization on your chunk and store the counts for each pre-token
        text_splits = re.split("|".join([token.replace("|", "\\|") for token in special_tokens]), chunk)
        for text_i in text_splits:
            # if text_i == "|":
            #     continue
            for s in re.finditer(PAT, text_i):
                s = s.group(0)
                s = tuple([bytes([i]) for i in s.encode("utf-8")])
                freq[s] = freq.get(s, 0) + 1
    return freq

def train_bpe(
    input_path: str, 
    vocab_size: int,
    special_tokens: list[str], 
    num_processes: int=8
    ):

    ### read text from input_path
    # with open(input_path, 'r') as file:
    #     text = file.read()
    

    ### initialize vocabulary
    vocab = {i: x.encode("utf-8") for i, x in enumerate(special_tokens)}
    vocab.update({i+len(special_tokens): bytes([i]) for i in range(256)})
    len_vocab = len(vocab)
    print("Initial vocabulary size", len_vocab)


    ### pre tokenization
    #### multiprocessing
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes*100, "<|endoftext|>".encode("utf-8"))

    pool = multiprocessing.Pool(num_processes)
    processes = [pool.apply_async(pretokenize_chunk, args=(start, end, input_path, special_tokens)) for start, end in zip(boundaries[:-1], boundaries[1:])]
    freq = {}
    for p in processes:
        freq_chunk = p.get()
        for s in freq_chunk:
            freq[s] = freq.get(s, 0) + freq_chunk[s]
    print("Pre-tokenization finished.")


    ### count frequency of every successive pair of bytes
    cnt_bytes_pairs = {}
    appear_bytes_pairs = {}
    for s in freq:
        if len(s) == 1:
            continue # nothing to merge
        bytes_pair_to_index = {}
        for j in range(len(s)-1):
            bytes_pair = (s[j], s[j+1])
            bytes_pair_to_index[bytes_pair] = bytes_pair_to_index.get(bytes_pair, []) + [j]
        for bytes_pair in bytes_pair_to_index:
            cnt_bytes_pairs[bytes_pair] = cnt_bytes_pairs.get(bytes_pair, 0) + freq[s]*len(bytes_pair_to_index[bytes_pair])
            appear_bytes_pairs[bytes_pair] = {**appear_bytes_pairs.get(bytes_pair, {}), **{s: bytes_pair_to_index[bytes_pair]}}

    print("Count frequency of every pair finished.")

    ### merge
    merges = []

    while len_vocab < vocab_size:

        if len_vocab % 100 == 0:
            print(f"Vocab size {len_vocab}")
        
        if len(cnt_bytes_pairs) == 0:
            break
        
        # print(cnt_bytes_pairs)
        # print(appear_bytes_pairs)

        ### find the pair with largest frequency
        merged_pair = max(cnt_bytes_pairs, key=lambda k: (cnt_bytes_pairs[k], k))
        if cnt_bytes_pairs[merged_pair] <= 0:
            break
        merges.append(merged_pair)
        vocab[len_vocab] = merged_pair[0]+merged_pair[1]
        len_vocab += 1
        # print("Merge", merged_pair)

        
        bytes_to_be_merged = appear_bytes_pairs[merged_pair].copy()
        for s in bytes_to_be_merged:

            ### update word-freq dict
            j = 0
            s_new = []
            while j <= len(s)-2:
                # print((curr[j], curr[j+1]))
                if (s[j], s[j+1]) == merged_pair: ## merge! 
                    s_new.append(s[j]+s[j+1])
                    j += 2
                else:
                    s_new.append(s[j])
                    j += 1
            if j == len(s)-1:
                s_new.append(s[j])
            s_new = tuple(s_new)

            freq[s_new] = freq.get(s_new, 0) + freq[s]
            # print(s, s_new)
            # print(s, indices, merged_pair)
            
            ### update pair-appearance/freq dict
            # delete all pairs in `s`
            for j in range(len(s)-1):
                bytes_pair = (s[j], s[j+1])
                if s in appear_bytes_pairs[bytes_pair]:
                    del appear_bytes_pairs[bytes_pair][s]
                cnt_bytes_pairs[bytes_pair] -= freq[s]

            # add all pairs in `s_new`
            bytes_pair_to_index = {}
            for j in range(len(s_new)-1):
                bytes_pair = (s_new[j], s_new[j+1])
                bytes_pair_to_index[bytes_pair] = bytes_pair_to_index.get(bytes_pair, []) + [j]
            for bytes_pair in bytes_pair_to_index:
                cnt_bytes_pairs[bytes_pair] = cnt_bytes_pairs.get(bytes_pair, 0) + freq[s_new]*len(bytes_pair_to_index[bytes_pair])
                appear_bytes_pairs[bytes_pair] = {**appear_bytes_pairs.get(bytes_pair, {}), **{s_new: bytes_pair_to_index[bytes_pair]}}

            del freq[s]

        # print(freq)
        del cnt_bytes_pairs[merged_pair]

    print("Final vocabulary size", len_vocab)
    return vocab, merges



if __name__ == "__main__":

    input_path = "/home/ec2-user/data/TinyStoriesV2-GPT4-train.txt"
    # "/home/ec2-user/assignment1-basics/tests/fixtures/corpus.en"
    # "/home/ec2-user/data/simple_text.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    start = time.time()
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens, num_processes=8)
    end = time.time()

    # print(merges)
    # print(vocab)
    print("Total time taken", end-start)

    result_path = "/home/ec2-user/bpe-tokenizer/"
    with open(f'{result_path}merges.txt', 'w') as f:
        for line in merges:
            f.write(f"{line}\n")
    with open(f'{result_path}vocab.json', 'w') as file:
        json.dump({k: str(vocab[k]) for k in vocab}, file)

    longest_token_id = max(vocab, key=lambda k: len(vocab[k]))
    print("Longest token", vocab[longest_token_id])