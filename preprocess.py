# -*- coding: utf-8 -*-
import os
import numpy
import pickle
import random
import argparse
from driver.Config import Configurable
from driver.MyIO import read_word_line
from driver.Vocab import VocabSrc, VocabTgt


def analysis(sentence_length, feature_dict=None, label_dict=None):

    if feature_dict is not None:
        print('单词个数为: ', len(feature_dict))
    if label_dict is not None:
        print('标签个数有: {0}个'.format(len(label_dict)))
        print('标签有: ')
        for k in label_dict.keys():
            print("标签为：{0}， 个数为：{1}".format(k, label_dict[k]))
    print()
    sentence_length = sorted(sentence_length.items(), key=lambda item: item[0], reverse=False)
    # sentence_length = sentence_length.most_common()
    count = 0
    for item in sentence_length:
        print("句子长度为： {0}  有{1}句".format(item[0], item[1]))
        count += int(item[1])
    print('句子个数为：', count)


if __name__ == '__main__':
    # random
    random.seed(666)
    numpy.random.seed(666)

    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='default.ini')
    parser.add_argument('--thread', type=int, default=1)
    args, extra_args = parser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    # data and analysis
    print('\n')
    train_data, train_sentence_len, feature_dict, label_dict = read_word_line(
        config.train_file, is_train=True)
    analysis(train_sentence_len, feature_dict, label_dict)
    if config.dev_file:
        print('\n')
        dev_data, dev_sentence_len = read_word_line(config.test_file)
        analysis(dev_sentence_len)
    print('\n')
    test_data, test_sentence_len = read_word_line(config.test_file)
    analysis(test_sentence_len)

    if not os.path.isdir(config.save_dir):
        os.mkdir(config.save_dir)
    pickle.dump(train_data, open(config.train_pkl, 'wb'))
    if config.dev_file:
        pickle.dump(dev_data, open(config.dev_pkl, 'wb'))
    pickle.dump(test_data, open(config.test_pkl, 'wb'))

    # vocab
    feature_list = [ite for ite, it in feature_dict.most_common(config.vocab_size)]
    label_list = [ite for ite in label_dict.keys()]
    pickle.dump(feature_list, open(config.save_feature_voc, 'wb'))
    pickle.dump(label_list, open(config.save_label_voc, 'wb'))
    feature_voc = VocabSrc(feature_list)
    label_voc = VocabTgt(label_list)

    # embedding
    if config.embedding_file != '' and len(config.embedding_file) != 0:
        embedding = feature_voc.create_vocab_embs(config.embedding_file)
        pickle.dump(embedding, open(config.embedding_pkl, 'wb'))
