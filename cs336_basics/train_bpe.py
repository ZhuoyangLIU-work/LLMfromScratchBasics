import regex as re
import os
import cProfile
import itertools
import sys
sys.path.append("/zfs/home/users/zyliu/RubymineProjects/LLMfromScratchBasicsImplement/cs336_basics")
from pretokenization_example import find_chunk_boundaries
# alternative, use relative path with from .pretokenization_example import find_chunk_boundaries
import multiprocessing as mp
from typing import BinaryIO
from tqdm import tqdm
from collections import Counter
SPECIAL_TOKENS = '<|endoftext|>'
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def chunk_pretokenize(
        file_path: str,
        start: int,
        next_start: int,
        special_tokens: list
) -> dict:
    '''
    This function pre-tokenizes a designated chunk of a file of bytestring.
    :param file: file handle to read the chunk of interest from, the source file contains a bytestring
    :param start: the start position at the file for the chunk of interest
    :param next_start: the end position + 1 at the file for the chunk of interest, i.e the chunk of interest is file_full_string[start:next_start]
    : param special_tokens: special tokens that need to be excluded when pre-tokenizating, for example, [<|endoftext|>]
    :return: Counter storing the pre-tokenization result
    '''
    # read the designated chunk from start to before next start
    with open(file_path, "rb") as file:
        file.seek(start)
        full_chunk = file.read(next_start-start).decode('utf-8', errors='ignore')

    # strip out all special tokens
    escaped = [re.escape(tok) for tok in special_tokens]
    cleaned_chunk = re.split('|'.join(escaped), full_chunk) if len(special_tokens)>0 else [full_chunk]

    # use PAT to map out pre-tokenization vocabulary
    pieces_matches = [re.finditer(PAT, piece) for piece in cleaned_chunk]
    chunk_pretoken_counter = Counter()
    for piece_matches in pieces_matches:
        for match in piece_matches:
            token = match.group(0)
            chunk_pretoken_counter[token] += 1
    return chunk_pretoken_counter


def pre_tokenize(
    file_path: str,
    num_processes: int,
    desirable_num_chunks: int,
    special_tokens: list = ['<|endoftext|>']
) -> Counter:

    # find boundaries to split the data feeding to each thread
    with open(file_path, "rb") as file:
        data_chunk_boundaries = find_chunk_boundaries(file, desirable_num_chunks, special_tokens[0].encode('utf-8'))
    tasks = [(file_path, start, end, special_tokens) for start, end in zip(data_chunk_boundaries[:-1], data_chunk_boundaries[1:])]
    pretoken_counts = Counter()
    if num_processes == 1:
        for task in tqdm(tasks):
            chunk_pretoken_counter = chunk_pretokenize(*task)
            pretoken_counts.update(chunk_pretoken_counter)
    else:
        with mp.Pool(num_processes) as pool:
            with tqdm(total=len(tasks)) as pbar:
                def update_pretoken_dict(chunk_counts: Counter) -> None:
                    pretoken_counts.update(chunk_counts)
                    pbar.update(1)

                for task in tasks:
                    pool.apply_async(chunk_pretokenize, args = task, callback = update_pretoken_dict)
                pool.close()
                pool.join()
    return pretoken_counts

def BPE_merge_one_step(
        freq_counter:Counter,
        merge_threshold:int,
) -> (Counter, tuple | None):
    '''
    Merge the most frequent consecutive pair of tokens
    :param freq_counter: current token tuples and their frequencies
    :param merge_threshold: threshold for merging
    :return: freq_counter, updated token tuples, and their frequencies
             merge_tuple, flagging which two consecutive pair of tokens are merged. If none, return None
    '''
    

def BPE_train_from_pretoken_counts(
    pretoken_counts: Counter
)-> (dict[int, bytes], list[tuple[bytes, bytes]]):
    '''
    Turns the pretokenization results into BPE tokens.
    :param pretoken_counts: Counter object counting occurence of each pretoken bytestring
    :return: vocab, dict[int, bytes], the tokenizer vocabulary, mapping from int (token ID in the vocabulary) to bytes (token bytes);
             merges, list[tuple[bytes, bytes]], a list of BPE merges produced from training. Each list item is a typle of bytes (<token1>, <token2>),
                        representing that <token1> was merged with <token2>. The merges should be ordered by the order of creation.
    '''
    token_frequencies = Counter({tuple(k) for k,v in pretoken_counts.items()})
    merge_threshold = token_frequencies.most_common(1)[0][1]



