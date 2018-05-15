# -*- coding: utf-8 -*-
import os
import torch
import numpy
import random
import argparse
from driver.MyIO import read_pkl
from driver.Vocab import VocabSrc, VocabTgt, PAD
from driver.Config import Configurable
from driver.Train import train
from TorchNN import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["OMP_NUM_THREADS"] = "1"


if __name__ == '__main__':
    # random
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    random.seed(666)
    numpy.random.seed(666)

    # gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: ", torch.backends.cudnn.enabled)

    # parameters
    parse = argparse.ArgumentParser('NER')
    parse.add_argument('--config_file', default='default.ini', type=str)
    parse.add_argument('--thread', default=1, type=int)
    parse.add_argument('--use_cuda', action='store_true', default=False)
    args, extra_args = parse.parse_known_args()

    config = Configurable(args.config_file, extra_args)
    torch.set_num_threads(args.thread)
    config.use_cuda = False
    if gpu and args.use_cuda:
        config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    # load data
    train_data = read_pkl(config.train_pkl)
    dev_data = read_pkl(config.dev_pkl)
    test_data = read_pkl(config.test_pkl)

    feature_list = read_pkl(config.load_feature_voc)
    label_list = read_pkl(config.load_label_voc)
    feature_voc = VocabSrc(feature_list)
    label_voc = VocabTgt(label_list)

    embedding = None
    if os.path.isfile(config.embedding_pkl):
        embedding = read_pkl(config.embedding_pkl)

    # model
    model = BILSTM(config, feature_voc.size, embedding[1] if embedding else config.embed_dim,
                   label_voc.size, PAD, embedding[0])

    # train
    train(model, train_data, dev_data, test_data, feature_voc, label_voc, config)
































