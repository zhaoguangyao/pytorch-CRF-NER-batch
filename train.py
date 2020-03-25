# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import random
import argparse
from TorchNN import *
from driver.Config import Configurable
from driver.IO import read_pkl
from driver.Vocab import PAD, VocabSrc, VocabTgt
from driver.Trainer import train


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"


if __name__ == '__main__':
    # random
    seed = 666
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    torch.backends.cudnn.enabled = False
    print("CuDNN: ", torch.backends.cudnn.enabled)

    # parameters
    parse = argparse.ArgumentParser('Attention Target Classifier')
    parse.add_argument('--config_file', type=str, default='default.ini')
    parse.add_argument('--thread', type=int, default=1)
    parse.add_argument('--use_cuda', action='store_true', default=True)
    args, extra_args = parse.parse_known_args()

    config = Configurable(args.config_file, extra_args)
    torch.set_num_threads(args.thread)
    config.use_cuda = False
    if gpu and args.use_cuda:
        config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    # load data
    train_data = read_pkl(config.train_pkl)
    dev_data = None
    if config.dev_file:
        dev_data = read_pkl(config.dev_pkl)
    test_data = read_pkl(config.test_pkl)

    feature_list = read_pkl(config.feature_voc)
    feature_voc = VocabSrc(feature_list)

    label_list = read_pkl(config.label_voc)
    label_voc = VocabTgt(label_list)

    embedding = None
    if os.path.isfile(config.embedding_pkl):
        embedding = read_pkl(config.embedding_pkl)

    # model
    model = BiLSTM_NCRFPP(config, feature_voc.size,
                embedding[1] if embedding else config.embed_dim,
                PAD, label_voc.size + 2,
                embedding[0] if embedding else None)
    if config.use_cuda:
        model = model.cuda()

    # train
    train(model, train_data, dev_data, test_data, feature_voc, label_voc, config)
