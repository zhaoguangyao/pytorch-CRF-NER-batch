# -*- coding: utf-8 -*-
import os
import numpy as np
from torch.utils.data import Dataset


def convert2numpy(feature, max_len):
    """
    将词序列映射为id序列

    Args:
        items: list, 词序列
        voc: item -> id的映射表
        max_len: int, 实例的最大长度

    Returns:
        arr: np.array, shape=[max_len,]
    """
    arr = np.zeros((max_len,), dtype='int64')
    min_range = min(max_len, len(feature))
    for i in range(min_range):
        arr[i] = feature[i]
    return arr


class SentenceDataset(Dataset):

    def __init__(self, data, max_len, vocab_src, vocat_tgt):
        """
        Args:
            max_len: int, 句子最大长度
            feature2id_dict: dict, 特征->id映射字典
        """
        self.data = data
        self.max_len = max_len
        self.vocab_src = vocab_src
        self.vocat_tgt = vocat_tgt

    def get_feature_dict(self, idx):
        data_idx = self.data[idx]
        feature = data_idx[0]
        label = data_idx[1]
        feature = self.vocab_src.word2id(feature)
        label = self.vocat_tgt.word2id(label)
        feature = convert2numpy(feature, self.max_len)
        label = convert2numpy(label, self.max_len)
        return (feature, label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.get_feature_dict(index)
