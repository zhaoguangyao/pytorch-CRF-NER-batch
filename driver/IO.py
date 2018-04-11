# -*- encoding: utf-8 -*-
import codecs
import pickle
from collections import Counter


def read_sentence_line(path):
    """
    读取一行是一句话的语料,比如情感分类
    Args:
        path: str, 语料文件路径
    Return:
        data: list中tuple
    """
    data = []
    with open(path, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            line = line.strip()
            if len(line) == 0 or line == '':
                print("an empty sentence, please check")
            else:
                data.append(line)
    return data


def read_word_line(path, is_train=False):
    """
    读取一个单词是一行的语料,比如NER
    如果读取的是train集，需要建立好词频的表，标签
    Args:
        path: str: 语料文件路径
        is_train: 判断是否是train集
    Return:
        data: list
        sentence_len: dict
        feature_dict: dict
        label_dict: dict
    """
    data = []
    sentence_len = Counter()
    feature_dict = Counter()
    label_dict = Counter()
    with open(path, 'r', encoding='utf-8') as input_file:
        sentence = []
        label = []
        for line in input_file:
            line = line.strip()
            if len(line) == 0 or line == '':
                data.append((sentence, label))
                sentence_len[len(sentence)] += 1
                sentence = []
                label = []
            else:
                strings = line.split(' ')
                sentence.append(strings[0])
                label.append(strings[1])
                if is_train:
                    feature_dict[strings[0]] += 1
                    label_dict[strings[1]] += 1
        if len(sentence) != 0:
            data.append((sentence, label))
    print('实例个数有: ', len(data))
    if is_train:
        return data, sentence_len, feature_dict, label_dict
    return data, sentence_len


# 为什么不用cvs来专门读取呢？？？
def read_csv(path, split=','):
    """
    读取csv文件

    Args:
        path: str, csv文件路径
        split: 分隔符号

    Return:
        terms: list
    """
    with open(path, 'r', encoding='utf-8') as file_csv:
        line = file_csv.readline()
        terms = []
        while line:
            line = line.strip()
            if not line:
                line = file_csv.readline()
                continue
            terms.append(line.split(split))
            line = file_csv.readline()
    return terms


def read_pkl(path):
    """
    读取pkl文件

    Args:
        path: str, pkl文件路径

    Return:
        pkl_ob: pkl对象
    """
    file_pkl = codecs.open(path, 'rb')
    return pickle.load(file_pkl)

