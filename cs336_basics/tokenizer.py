from collections import defaultdict

import numpy as np
import jax
from functools import partial
from cs336_basics.train_bpe import string_pretokenize, file_pretokenize, train_bpe, PAT
import json, yaml, os
import regex as re
from collections.abc import Iterable



class Tokenizer:
    '''
    Given a vocabulary and a list of merges, encodes text into integer IDs and decodes integer IDs into text.
    Support user-provided special tokens (appending them to the vocabulary if they aren’t already there)
    ----------------------------------------------------------------------------------------------------
    Attributes:
        :attr self._vocab
        :attr self._merges
        :attr self._special_tokens
        :attr self._special_tokens_pattern
        :attr self._vocab_inv
        :attr self._merges_rank
        :attr self._cache_encode
    Methods:
        :method get_attrs
        :method _encode_single_pretoken
        :method _encode_chunk
        :method encode

    '''
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        '''

        :param vocab:
        :param merges:
        :param special_token:
        '''
        self._vocab = vocab
        self._merges = merges
        self._special_tokens = special_tokens if special_tokens else []
        if len(self._special_tokens) >= 2:
            self._special_tokens.sort(reverse=True)
        ### sorting in reverse lexical order to ensuring recognizing the longest special tokens without spliting from the middle
        # e.g.
        # special_tokens = ['<|endoftext|>', '<|endoftext|><|endoftext|>']
        # orginal text = Hello, how <|endoftext|><|endoftext|> are you? <|endoftext|>
        self._special_tokens_pattern = '|'.join(
            [re.escape(tok) for tok in special_tokens]) if special_tokens else ''
        self._vocab_inv = {v: k for k, v in vocab.items()}
        self._merges_rank = {merges[idx]: idx for idx in range(len(merges))}
        self._cache_encode = {}  # dict[str, list[int] that matches pretoken strings into corresponding lists of token IDs

    def get_attrs(self, attr: str):
        attr_dict = {
            'vocab': self._vocab,
            'merges': self._merges,
            'special_tokens': self._special_tokens,
            'special_tokens_pattern': self._special_tokens_pattern,
            'vocab_inv': self._vocab_inv,
            'merges_rank': self._merges_rank,
        }
        return attr_dict.get(attr, None)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        def load_mapping(path: str) -> dict:
            ext = os.path.splitext(path)[1].lower()
            with open(path, 'r', encoding='utf-8') as f:
                if ext in ('.yaml', '.yml'):
                    return yaml.safe_load(f)
                elif ext == '.json':
                    return json.load(f)
                else:
                    raise ValueError(f"Unsupported file extension for {path}")

        # 1) Load and decode vocab: { str(id): hexstring } → { int: bytes }
        raw_vocab = load_mapping(vocab_filepath)
        vocab = {int(k): bytes.fromhex(v) for k, v in raw_vocab.items()}

        # 2) Load and decode merges: [ [hex1,hex2], ... ] → [ (bytes, bytes), ... ]
        raw_merges = load_mapping(merges_filepath)
        merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in raw_merges]

        return cls(vocab, merges, special_tokens)


    def _encode_single_pretoken(self, pretoken_repr: tuple[bytes]) -> tuple[bytes]:
        '''
        merge the provided pretoken according to the merge order provided in self._merges
        :param pretoken_repr:
        :return:
        '''
        while True:
            best_rank , best_idx = np.inf, None

            # scan through the current repr for potential merges and store the top ranked
            for i in range(len(pretoken_repr)-1):
                rank = self._merges_rank.get((pretoken_repr[i], pretoken_repr[i+1]), np.inf)

                if rank < best_rank:
                    best_rank = rank
                    best_idx = i

            if best_idx == None:
                return pretoken_repr

            new_repr = pretoken_repr[:best_idx] + (pretoken_repr[best_idx] + pretoken_repr[best_idx+1],) + pretoken_repr[best_idx+2:]
            pretoken_repr = new_repr

    def _encode_chunk(self, chunk: str) -> list[int]:
        '''
        encode a chunk of string free of special tokens
        :param chunk: a string of text to encode that does not contain any special tokens
        :return:
        '''
        if not str: return []
        matches = re.finditer(PAT, chunk)
        chunk_encoded = []
        for match in matches:
            pretoken_str = match.group(0)
            pretoken_encoded = self._cache_encode.get(pretoken_str, None)
            if pretoken_encoded == None: # if not already in cache
                pretoken_repr = self._encode_single_pretoken(tuple(bytes([b]) for b in pretoken_str.encode("utf-8")))
                pretoken_encoded = [self._vocab_inv[word] for word in pretoken_repr] # find the token IDs
                self._cache_encode[pretoken_str] = pretoken_encoded

            chunk_encoded.extend(pretoken_encoded)
        return chunk_encoded

    def encode(self, text: str) -> list[int]:
        '''
        Encode an input text into a sequence of token IDs. The input can include special tokens.
        :param text:
        :return:
        '''
        # 0) initialization
        ids = []
        # 1) split string with special tokens
        if not self._special_tokens:
            return self._encode_chunk(text)
        chunks = re.split('('+self._special_tokens_pattern+')', text)
        ### Use of capturing parenthesis in re.split:
        # s = 'abccccccc<|endoftext|>ddddd'
        # special_tokens = ['<|endoftext|>']
        # p = '|'.join([re.escape(tok) for tok in special_tokens])
        # re.split(p, s)
        # ['abccccccc', 'ddddd']
        # re.split('(' + p + ')', s)
        # ['abccccccc', '<|endoftext|>', 'ddddd']

        # 2) pretokenize chunks and add into final encoding
        for chunk in chunks:
            if chunk.encode('utf-8') in self._vocab_inv:
                ids.append(self._vocab_inv[chunk.encode('utf-8')])
            else:
                chunk_encoding = self._encode_chunk(chunk)
                ids.extend(chunk_encoding)
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> list[int]:
        '''
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into  memory
        :param iterable:
        :return:
        '''
        for iter_piece in iterable:
            yield from self.encode(iter_piece)

    def decode(self, ids: list[int]) -> str:
        '''
        Decode a sequence of token IDs into text
        :param ids:
        :return:
        '''
        text_out = b''.join([self._vocab[id] for id in ids]).decode('utf-8', errors='replace')
        return text_out



