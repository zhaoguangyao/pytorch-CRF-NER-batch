# -*- encoding: utf-8 -*-
import re
import codecs
import pickle
from collections import Counter


def analysis(sentence_length, words=None, label=None):
    if words is not None:
        print('单词个数为：', len(words))
    if label is not None:
        print('标签个数有：{0}个'.format(len(label)))
        print('标签有：')
        for i in label.keys():
            print("标签为：{0}，个数有：{1}".format(i, label[i]))
    sentence_length = sorted(sentence_length.items(), key=lambda k: k[0], reverse=False)
    # sentence_length = sentence_length.most_common()
    count = 0
    for item in sentence_length:
        print("句子长度为：{0}，有{1}".format(item[0], item[1]))
        count += item[1]
    print("句子个数为：", count)


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = string.lower()
    string = re.sub(r"，", ",", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)

    string = re.sub(r"^-user-$", "<user>", string)
    string = re.sub(r"^-url-$", "<url>", string)
    string = re.sub(r"^-lqt-$", "\'", string)
    string = re.sub(r"^-rqt-$", "\'", string)
    # 或者
    # string = re.sub(r"^-lqt-$", "\"", string)
    # string = re.sub(r"^-rqt-$", "\"", string)
    string = re.sub(r"^-lrb-$", "\(", string)
    string = re.sub(r"^-rrb-$", "\)", string)
    string = re.sub(r"^lol$", "<lol>", string)
    string = re.sub(r"^<3$", "<heart>", string)
    string = re.sub(r"^#.*", "<hashtag>", string)
    string = re.sub(r"^[0-9]*$", "<number>", string)
    string = re.sub(r"^\:\)$", "<smile>", string)
    string = re.sub(r"^\;\)$", "<smile>", string)
    string = re.sub(r"^\:\-\)$", "<smile>", string)
    string = re.sub(r"^\;\-\)$", "<smile>", string)
    string = re.sub(r"^\;\'\)$", "<smile>", string)
    string = re.sub(r"^\(\:$", "<smile>", string)
    string = re.sub(r"^\)\:$", "<sadface>", string)
    string = re.sub(r"^\)\;$", "<sadface>", string)
    string = re.sub(r"^\:\($", "<sadface>", string)
    return string.strip()


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
    英语
    读取一个单词是一行的语料,比如NER
    如果读取的是train集，需要建立好词频的表，标签
    Args:
        path: str, 语料文件路径
        is_train: boolean, 判断是否是train集
    Return:
        data: list
        sentence_len: dict
        feature_dict: dict
        label_dict: dict
    """
    data = []
    sentence_len = Counter()
    word_dict = Counter()
    label_dict = Counter()
    with open(path, 'r', encoding='utf-8') as input_file:
        words = []
        labels = []
        for line in input_file:
            line = line.strip()
            if len(line) == 0 or line == '':
                sentence_len[len(words)] += 1
                data.append((words, labels))
                words = []
                labels = []
            else:
                strings = line.split(' ')
                if len(strings) != 2:
                    print("金标有问题")
                    exit(0)
                words.append(strings[0].lower())
                labels.append(strings[-1])
                word_dict[strings[0].lower()] += 1
                label_dict[strings[-1]] += 1

        if len(words) != 0:
            data.append((words, labels))
            sentence_len[len(words)] += 1
    print('实例个数有: ', len(data))
    analysis(sentence_len, word_dict, label_dict)
    if is_train:
        return data, word_dict, label_dict
    return data


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

