from typing import Union
from typing import Iterable, Iterator
import ast
import json
import regex as re
import numpy as np
import pickle
import time

class tokenizer():

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: Union[list[str], None] = None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        """
        self.merges = merges
        self.vocab = vocab
        self.special_tokens = special_tokens if type(special_tokens) == list else []
        self.special_tokens_set = set(special_tokens if type(special_tokens) == list else [])
        vocab_size = len(self.vocab)
        if special_tokens is not None and len(special_tokens) > 0:
            for special_token in special_tokens:
                # print(special_token)
                if special_token.encode("utf-8") not in vocab.values():
                    self.vocab[vocab_size] = special_token.encode("utf-8")
                    vocab_size += 1
        # print(self.vocab)
        self._bytes_to_id_dict = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Union[list[str], None] = None):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special tokens.  
        """
        
        ## read merges
        with open(merges_filepath, 'r') as f:
            merges_lines = f.readlines()
        merges = []
        for line in merges_lines:
            merge_pair = ast.literal_eval(line.rstrip("\n"))
            merges.append(merge_pair)
        
        ## read vocab
        with open(vocab_filepath) as f:
            vocab_read = json.load(f)
        vocab = {int(vocab_index): ast.literal_eval(vocab_item) for vocab_index, vocab_item in vocab_read.items()}
        
        obj = cls.__new__(cls)
        obj.__init__(vocab=vocab, merges=merges, special_tokens=special_tokens)
        return obj
    
    def _split_by_speical_tokens(self, text: str) -> list[str]:

        ## handle special tokens here
        if len(self.special_tokens) > 0:
            special_tokens_reorder = sorted(self.special_tokens, key=len, reverse=True)
            special_token_pat = "(" + "|".join([token.replace("|", "\\|") for token in special_tokens_reorder]) + ")"
            # print("special_token_pat", special_token_pat)
            text_split_special_token = re.split(special_token_pat, text)
            # print("text after split by special tokens", text_split_special_token)
        else:
            text_split_special_token = [text]
        
        return text_split_special_token
    
    def _apply_merges_to_pre_token(self, pre_token: bytes) -> tuple[bytes]:
        
        curr = tuple([bytes([i]) for i in pre_token])
        new = []
        for merge_pair in self.merges:

            j = 0
            while j <= len(curr)-2:
                # print((curr[j], curr[j+1]))
                if (curr[j], curr[j+1]) == merge_pair: ## merge! 
                    new.append(curr[j]+curr[j+1])
                    j += 2
                else:
                    new.append(curr[j])
                    j += 1
            if j == len(curr)-1:
                new.append(curr[j])

            curr = tuple(new)
            new = []

            if len(curr) == 1: ## nothing to merge
                break
        
        # print("Input", pre_token)
        # print("Output", curr)
        # print()
        return curr
        
    def encode(self, text: str) -> list[int]: 
        """
        Encode an input text into a sequence of token IDs.
        """

        text_split_special_token = self._split_by_speical_tokens(text)
        
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        encode_ids = []
        for text_i in text_split_special_token:
            if text_i in self.special_tokens_set: ## it is a special token
                encode_ids.append(self._bytes_to_id_dict.get(text_i.encode("utf-8")))
            else:
                for s in re.finditer(PAT, text_i): ## pre-token
                    pre_token = s.group(0).encode("utf-8")
                    merged_pre_token = self._apply_merges_to_pre_token(pre_token)
                    for b in merged_pre_token:
                        encode_ids.append(self._bytes_to_id_dict.get(b))

        return encode_ids

    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory.
        """
        for text in iterable:
            encode_ids = self.encode(text)
            for id in encode_ids:
                yield id
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        bytes_seq = [self.vocab.get(id) for id in ids]
        return b"".join(bytes_seq).decode("utf-8", errors='replace')



if __name__ == "__main__":

    # tokenizer_bpe = tokenizer(
    #     vocab={0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}, 
    #     merges=[(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')], 
    #     special_tokens=["<|endoftext|>", "heyyyy", "<|endoftext|><|endoftext|>"]
    # )

    # input_text = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    # input_text = "heyyyy, Here is some text i'd like to encode <|endoftext|>"
    # input_text = "the cat ate"

    # encode_ids = tokenizer_bpe.encode(input_text)
    # print(encode_ids)
    # output_text = tokenizer_bpe.decode(ids=encode_ids)
    # print(input_text == output_text)



    bpe_train_path = "/home/ec2-user/bpe-tokenizer/"
    tokenizer_bpe = tokenizer.from_files(
        vocab_filepath=f'{bpe_train_path}vocab.json', 
        merges_filepath=f'{bpe_train_path}merges.txt', 
        special_tokens=["<|endoftext|>"]
    )

    for dataset in ["valid", "train"]:
        input_text_path = f"/home/ec2-user/data/TinyStoriesV2-GPT4-{dataset}.txt"
        # with open(input_text_path, 'r') as file:
        #     input_text = file.read()
        # print(f"Text read from {input_text_path}")
        file = open(input_text_path, 'r')
        
        start = time.time()
        # encode_ids = tokenizer_bpe.encode(input_text)
        encode_ids_iterator = tokenizer_bpe.encode_iterable(file)
        encode_ids = []
        for id in encode_ids_iterator:
            encode_ids.append(id)
        encode_ids_np = np.array(encode_ids, dtype=np.uint16)
        print("Token count", encode_ids_np.shape[0])

        output_filename = f"/home/ec2-user/data/TinyStoriesV2-GPT4-{dataset}-encoded-iterable.npy"
        np.save(output_filename, encode_ids_np)
        end = time.time()

        print("Total time taken", end-start)


    # with open(output_filename, 'wb') as file:
    #     pickle.dump(encode_ids_np, file)