# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch.autograd import Variable


def create_batch_iter(data, batch_size=1, shuffle=False):
    if shuffle:
        np.random.shuffle(data)

    batched_data = []
    instances = []
    for instance in data:
        instances.append(instance)
        if len(instances) == batch_size:
            batched_data.append(instances)
            instances = []

    if len(instances) > 0:
        batched_data.append(instances)

    for batch in batched_data:
        yield batch


def pair_data_variable(batch, vocab_srcs, vocab_tgts, config):
    batch_size = len(batch)
    batch = sorted(batch, key=lambda b: len(b[0]), reverse=True)
    src_lengths = [len(batch[i][0]) for i in range(batch_size)]
    max_src_length = int(src_lengths[0])

    src_words = Variable(torch.LongTensor(max_src_length, batch_size).zero_(), requires_grad=False)
    tgt_words = Variable(torch.LongTensor(max_src_length, batch_size).zero_(), requires_grad=False)

    for idx, instance in enumerate(batch):
        sentence = vocab_srcs.word2id(instance[0])
        for idj, value in enumerate(sentence):
            src_words.data[idj][idx] = value
        label = vocab_tgts.word2id(instance[1])
        for idz, value in enumerate(label):
            tgt_words.data[idz][idx] = value

    if config.use_cuda:
        src_words = src_words.cuda()
        tgt_words = tgt_words.cuda()

    return src_words, tgt_words, src_lengths
