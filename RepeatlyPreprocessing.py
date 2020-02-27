import random
import numpy as np
import torch

"""
Preprocessing for NMT vs ASR toy experiments
Idea: Make NMT tasks with modified inputs like ASR inputs (randomly repeated words + noise) 
1. Repeat each input words 2 to 6 times
2. Add Gaussian Noise on the inputs before training
3. Use Truncated attention with (L, R) and Universal Attention
(4?) Duplicated Area(Add Several Word Vectors)
(5?) Insert Epsilon Randomly
6. Use SpecAugment for NMT
"""


def repeat_word(seq, num_repeat_words, repeat_range):
    new_seq = []
    if num_repeat_words is None:
        num_repeat_words = []
        for word in seq:
            if repeat_range > 0:
                num = np.random.randint(repeat_range)+1
            else:
                num = 1
            num_repeat_words.append(num)
            for _ in range(num):
                new_seq.append(word)

            # # (5?) Insert Epsilon Randomly
            # if np.random.random() < 0.2:
            #     for _ in range(np.random.randint(8)+1):
            #         new_seq.append(3)

    else:
        for i, word in enumerate(seq):
            num = num_repeat_words[i]
            for _ in range(num):
                new_seq.append(word)

    return new_seq, len(new_seq), num_repeat_words


def repeat_input_words(src_batch, num_repeat_words, repeat_range):
    # Do preprocessing #1
    if num_repeat_words == -1:
        num_repeat_words = None
    num_repeat_words_list = []
    new_seqs = []
    max_len = 0
    for i, seq in enumerate(src_batch):
        seq.append(2)
        seq_strip = seq[:seq.index(2)]
        if num_repeat_words is None:
            new_seq, new_len, num_repeat_word = repeat_word(seq_strip, num_repeat_words, repeat_range)
        else:
            new_seq, new_len, num_repeat_word = repeat_word(seq_strip, num_repeat_words[i], repeat_range)
        new_seqs.append(new_seq)
        num_repeat_words_list.append(num_repeat_word)
        if max_len < new_len:
            max_len = new_len

    seqs = []
    for new_seq in new_seqs:
        new_padded_seq = [2] * max_len
        new_padded_seq[:len(new_seq)] = new_seq
        seqs.append(new_padded_seq)

    return seqs, num_repeat_words_list
