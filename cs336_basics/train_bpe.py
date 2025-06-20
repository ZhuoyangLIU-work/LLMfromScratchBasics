import regex as re
import os
import cProfile
import itertools
import sys

from cs336_basics.pretokenization_example import find_chunk_boundaries_listspecials

sys.path.append("/zfs/home/users/zyliu/RubymineProjects/LLMfromScratchBasicsImplement/cs336_basics")
from pretokenization_example import find_chunk_boundaries
# alternative, use relative path with from .pretokenization_example import find_chunk_boundaries
import multiprocessing as mp
from typing import BinaryIO
from tqdm import tqdm
from collections import Counter


'''
Global variables
'''
SPECIAL_TOKENS = '<|endoftext|>'
MERGE_THRESHOLD = 0
VOCAB_SIZE = 10000
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def chunk_pretokenize(
        file_path: str,
        start: int,
        next_start: int,
        special_tokens: list = [SPECIAL_TOKENS]
) -> Counter[tuple[bytes, ...]]:
    '''
    This function pre-tokenizes a designated chunk of a file of bytestring.
    :param file: file handle to read the chunk of interest from, the source file contains a bytestring
    :param start: the start position at the file for the chunk of interest
    :param next_start: the end position + 1 at the file for the chunk of interest, i.e the chunk of interest is file_full_string[start:next_start]
    :param special_tokens: special tokens that need to be excluded when pre-tokenizating, for example, [<|endoftext|>]
    :return: Counter[tupe[bytes], int] storing the pre-tokenization result
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
            # Each character byte is wrapped into a separate 1-byte token for BPE granularity
            token = tuple(bytes([b]) for b in bytes(match.group(0).encode("utf-8")))
            chunk_pretoken_counter[token] += 1
    return chunk_pretoken_counter


def pre_tokenize(
    file_path: str,
    num_processes: int,
    desirable_num_chunks: int,
    special_tokens: list = [SPECIAL_TOKENS]
) -> Counter:
    '''
    Update: Currently tested for len(special_tokens) = 1
    :param file_path:
    :param num_processes:
    :param desirable_num_chunks:
    :param special_tokens:
    :return:
    '''

    # find boundaries to split the data feeding to each thread
    with open(file_path, "rb") as file:
        if len(special_tokens) == 1:
            data_chunk_boundaries = find_chunk_boundaries(file, desirable_num_chunks, special_tokens[0].encode('utf-8'))
        else:
            data_chunk_boundaries = find_chunk_boundaries_listspecials(file, desirable_num_chunks, [special_tokens[i].encode('utf-8') for i in range(len(special_tokens))])
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
    token_freq_counter: Counter[tuple[bytes], int],
    merge_threshold: int = MERGE_THRESHOLD
) -> (Counter, tuple | None):
    '''
    Merge the most frequent consecutive pair of tokens
    :param freq_counter: current token tuples and their frequencies
    :param merge_threshold: threshold for merging
    :return: freq_counter, updated token tuples, and their frequencies
             merge_tuple, flagging which two consecutive pair of tokens are merged. If none, return None
    '''
    pair_freq_counter = Counter()

    # find frequencies of all consecutive pairs of tokens
    for key in token_freq_counter:
        # print(f'    BPE_merge_one_step: counter_freq_for_loop:: key = {key}')
        for id in range(len(key)-1):
            pair_freq_counter[(key[id], key[id+1])] += token_freq_counter[key]
            # print(f'        BPE_merge_one_step: counter_freq_for_loop:: pair = {(key[id], key[id+1])}, freq = {pair_freq_counter[(key[id], key[id + 1])]}')

    if not pair_freq_counter:
        return token_freq_counter, None

    max_count = max(pair_freq_counter.values())
    # print(f'BPE_merge_one_step: pair_freq_counter={pair_freq_counter}')

    if max_count < merge_threshold:
        return token_freq_counter, None

    merge_candidates_list = [pair for pair, count in pair_freq_counter.items() if count == max_count]
    # print(f'BPE_merge_one_step: merge_candidates_list={merge_candidates_list}')

    merge_pair = max(merge_candidates_list)

    # print(f'----merge_pair={merge_pair}')

    # merge_pair_flat = tuple(itertools.chain.from_iterable(
    #                         [x if isinstance(x, tuple) else (x,) for x in merge_pair]
    #                         ))

    merge_pair_flat = merge_pair[0] + merge_pair[1]
    swaps = list()

    for key in token_freq_counter.keys():
        merged = []
        i = 0
        while i < len(key):
            if i < len(key) - 1 and key[i] == merge_pair[0] and key[i + 1] == merge_pair[1]:
                # print(f'merge_pair_flat={merge_pair_flat}')
                merged.append(merge_pair_flat)  # Concatenate into one token
                i += 2
            else:
                merged.append(key[i])
                i += 1

        # update the token_freq_counter with the merged version for keys
        swaps.append(tuple([tuple(merged), key]))
    for swap in swaps:
        token_freq_counter[swap[0]] = token_freq_counter.pop(swap[1])

    return token_freq_counter, merge_pair

def BPE_train_from_pretoken_counts(
    pretoken_counts: Counter[str, int],
    target_vocab_size: int = VOCAB_SIZE,
    special_tokens: list = [SPECIAL_TOKENS],
    merge_threshold: int = MERGE_THRESHOLD,
)-> (dict[int, bytes], list[tuple[bytes, bytes]]):
    '''
    Turns the pretokenization results into BPE tokens.
    :param pretoken_counts: Counter object counting occurence of each pretoken bytestring:
    :param target_vocab_size: target vocab size
    :param special_tokens: list of special tokens
    :return: vocab, dict[int, bytes], the tokenizer vocabulary, mapping from int (token ID in the vocabulary) to bytes (token bytes);
             merges, list[tuple[bytes, bytes]], a list of BPE merges produced from training. Each list item is a typle of bytes (<token1>, <token2>),
                        representing that <token1> was merged with <token2>. The merges should be ordered by the order of creation.
    '''
    token_frequencies = pretoken_counts
    vocab_list = [bytes([i]) for i in range(256)]
    vocab_list.extend([bytes(special_tokens[i].encode('utf-8')) for i in range(len(special_tokens))])
    vocab_dict = {i: item for i, item in enumerate(vocab_list)}
    position = len(vocab_list)
    merges = list()

    while len(vocab_dict) < target_vocab_size:
        # assert len(vocab_list) == position, 'BPE_train_from_pretoken_counts: merge results in duplicated tokenizer vocabulary.'
        token_frequencies, merge_tuple = BPE_merge_one_step(token_frequencies, merge_threshold)

        # Cannot further merge if tokens have reached full lengths of pretokenizer vocab
        if merge_tuple is None:
            break
        pair1 = merge_tuple[0]
        pair2 = merge_tuple[1]
        BPE_new_token = bytes(pair1) + bytes(pair2)
        vocab_list.append(BPE_new_token)
        vocab_dict[position] = BPE_new_token
        position += 1
        merges.append((pair1, pair2,))

    return vocab_dict, merges

def train_bpe(
        input_path: str,
        num_processes_pret: int = 1,
        num_chunks_pret: int = 1,
        vocab_size: int = VOCAB_SIZE,
        special_tokens: list[str] = [SPECIAL_TOKENS],
) -> (dict[int, bytes], list[tuple[bytes, bytes]]):
    '''
    Trains a BPE model, returning a vocabulary, mapping from int (token ID in the vocabulary) to the byte string, and the merges produced from training, in the same order in which they were merged.
    :param input_path: str, path to a text file with BPE tokenizer training data
    :param vocab_size: int, a positive integer that defines the maximum final vocabulary size
    :param special_tokens: list[str], a list of strings to add to the vocabulary. These special tokens do not otherwise affect BPE training
    :return vocab: dict[int, bytes], the tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes)
            mergers: list[tuple[bytes, bytes]], a list of BPE merges produced from training, in the order of creation.
    '''
    pretoken_counts = pre_tokenize(input_path, num_processes_pret, num_chunks_pret, special_tokens)
    vocab_dict, merges = BPE_train_from_pretoken_counts(pretoken_counts, vocab_size, special_tokens)
    return vocab_dict, merges


if '__name__' == '__main__':
    # initial_counter = {(b'l', b'o', b'w'): 5, (b'l', b'o', b'w',b'e',b'r'): 2, (b'w', b'i', b'd', b'e', b's', b't'): 3,  (b'n', b'e', b'w', b'e', b's', b't'): 6}
    pass




