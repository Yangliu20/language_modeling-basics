from typing import Union
from typing import Iterable, Iterator
import ast
import json


class tokenizer():

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: Union[list[str], None] = None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        """
        self.merges = merges
        self.vocab = vocab
        vocab_size = len(self.vocab)
        if special_tokens is not None and len(special_tokens) > 0:
            for special_token in special_tokens:
                print(special_token)
                if special_token not in vocab.values():
                    self.vocab[vocab_size] = special_token.encode("utf-8")
                    vocab_size += 1
        # print(self.vocab)

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
        
        
    def encode(self, text: str) -> list[int]: 
        """
        Encode an input text into a sequence of token IDs.
        """
        pass
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory.
        """
        pass
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        pass



if __name__ == "__main__":

    bpe_train_path = "/home/ec2-user/bpe-tokenizer/"

    tokenizer_bpe = tokenizer.from_files(
        vocab_filepath=f'{bpe_train_path}vocab.json', 
        merges_filepath=f'{bpe_train_path}merges.txt', 
        special_tokens=["heyyyy", "shahsjdjfh"]
    )