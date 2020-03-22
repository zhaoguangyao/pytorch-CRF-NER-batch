# -*- coding: utf-8 -*-
import torch
import numpy as np


def create_batch_iter(data, batch_size, shuffle=False):
    if shuffle:
        np.random.shuffle(data)

    batched_data = []
    instances = []
    count = 0
    for instance in data:
        instances.append(instance)
        count += 1
        if count == batch_size:
            batched_data.append(instances)
            instances = []
            count = 0
    if count != 0:
        batched_data.append(instances)
    for batch in batched_data:
        yield batch


def pair_data_variable(batch, vocab_srcs, vocab_tgts, use_cuda):
    batch_size = len(batch)
    batch = sorted(batch, key=lambda b: len(b[0]), reverse=True)
    src_lengths = [len(batch[i][0]) for i in range(batch_size)]
    max_src_length = int(src_lengths[0])

    src_words = torch.zeros([max_src_length, batch_size], dtype=torch.int64, requires_grad=False)
    tgt_words = torch.full([batch_size, max_src_length], vocab_tgts.size + 1, dtype=torch.int64, requires_grad=False)
    src_mask = torch.zeros([batch_size, max_src_length], dtype=torch.bool, requires_grad=False)

    # k = 0
    for idx, instance in enumerate(batch):
        words = vocab_srcs.word2id(instance[0])
        labels = vocab_tgts.word2id(instance[1])
        for idj, word in enumerate(words):
            src_words[idj][idx] = word
            src_mask[idx][idj] = 1
        for idj, label in enumerate(labels):
            tgt_words[idx][idj] = label
            # k += 1

    if use_cuda:
        src_words = src_words.cuda()
        tgt_words = tgt_words.cuda()
        src_mask = src_mask.cuda()

    return batch, src_words, tgt_words, src_lengths, src_mask
