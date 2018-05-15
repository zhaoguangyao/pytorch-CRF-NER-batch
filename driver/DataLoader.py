# -*- coding: utf-8 -*-
"""
本来很想用dataloader，但是显然她是不支持每句话不等长的处理方式的。
但是自己写的是不支持多线程加载数据的。

torchtext不知道是怎么封装的，到底有没有多线程???
torchtext一直都是一个心结啊，一定要看懂，最好能实现一个简化版本。
"""
import torch
import numpy as np
from torch.autograd import Variable


def pair_data_variable(batch, vocab_srcs, vocab_tgts, config):
    batch_size = len(batch)
    batch = sorted(batch, key=lambda b: len(b[0]), reverse=True)
    src_lengths = [len(batch[i][0]) for i in range(batch_size)]
    max_src_length = int(src_lengths[0])

    src_words = Variable(torch.LongTensor(batch_size, max_src_length).zero_(), requires_grad=False)
    tgt_words = Variable(torch.LongTensor(batch_size, max_src_length).zero_(), requires_grad=False)

    for idx, instance in enumerate(batch):
        sentence = vocab_srcs.word2id(instance[0])
        for idj, value in enumerate(sentence):
            src_words.data[idx][idj] = value
        label = vocab_tgts.word2id(instance[1])
        for idz, value in enumerate(label):
            tgt_words.data[idx][idz] = value

    if config.use_cuda:
        src_words = src_words.cuda()
        tgt_words = tgt_words.cuda()

    return src_words, tgt_words, src_lengths


# class DataLoader:
#     def __init__(self, data, vocab_src, vocat_tgt, batch_size=1, shuffle=False,
#                  num_workers=1):
#         self.data = data
#         self.vocab_src = vocab_src
#         self.vocab_tgt = vocat_tgt
#         self.batch_size = batch_size
#         self.shuffle = shuffle


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


# def convert2numpy(seq, max_len, is_train):
#     """
#     转化为numpy,把太长的句子切分一下
#
#     Args:
#         seq: list, 序列
#         max_len: int, 实例的最大长度
#
#     Returns:
#         arr: np.array, shape=[max_len,]
#     """
#     if len(seq) > max_len and (is_train is True):
#         arr = np.zeros((max_len,), dtype='int64')
#     else:
#         arr = np.zeros((len(seq),), dtype='int64')
#     min_range = min(max_len, len(seq))
#     for i in range(min_range):
#         arr[i] = seq[i]
#     return arr


# class Dataset:
#
#     def __init__(self, data, max_len, vocab_src, vocat_tgt, is_train=False):
#         """
#         Args:
#             max_len: int, 句子最大长度
#             feature2id_dict: dict, 特征->id映射字典
#         """
#         self.data = data
#         self.max_len = max_len
#         self.vocab_src = vocab_src
#         self.vocat_tgt = vocat_tgt
#         self.is_train= is_train
#
#     def get_by_idx(self, idx):
#         data_idx = self.data[idx]
#         feature = data_idx[0]
#         label = data_idx[1]
#
#         feature = self.vocab_src.word2id(feature)
#         label = self.vocat_tgt.word2id(label)
#
#         # feature = convert2numpy(feature, self.max_len, is_train=self.is_train)
#         # label = convert2numpy(label, self.max_len, is_train=self.is_train)
#         result = (feature, label)
#         return result
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         return self.get_by_idx(index)
