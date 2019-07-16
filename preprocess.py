# -*- coding: utf-8 -*-
import numpy
import pickle
import random
import argparse
from driver.config import Configurable
from driver.io import read_word_line
from driver.vocab import VocabSrc, VocabTgt


if __name__ == '__main__':
    # random
    random.seed(666)
    numpy.random.seed(666)

    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='./default.ini')
    parser.add_argument('--thread', type=int, default=1)
    args, extra_args = parser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    # data
    print('\n')
    train_data, feature_dict, label_dict = read_word_line(config.train_file, is_train=True)
    pickle.dump(train_data, open(config.train_pkl, 'wb'))

    if config.dev_file:
        print('\n')
        dev_data = read_word_line(config.dev_file, is_train=False)
        pickle.dump(dev_data, open(config.dev_pkl, 'wb'))

    print('\n')
    test_data = read_word_line(config.test_file, is_train=False)
    pickle.dump(test_data, open(config.test_pkl, 'wb'))

    # vocab
    feature_list = [k for k, v in feature_dict.most_common(config.vocab_size)]
    label_list = [k for k in label_dict.keys()]
    pickle.dump(feature_list, open(config.feature_voc, 'wb'))
    pickle.dump(label_list, open(config.label_voc, 'wb'))

    feature_voc = VocabSrc(feature_list)
    label_voc = VocabTgt(label_list)

    # embedding
    if config.embedding_file:
        embedding, embed_dim = feature_voc.create_vocab_embs(config.embedding_file)
        pickle.dump((embedding, embed_dim), open(config.embedding_pkl, 'wb'))
