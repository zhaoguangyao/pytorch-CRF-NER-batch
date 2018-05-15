# -*- encoding: utf-8 -*-
import codecs
import pickle
from collections import Counter


def read_word_line(path, is_train=False):
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


def read_pkl(path):
    file_pkl = codecs.open(path, 'rb')
    return pickle.load(file_pkl)

