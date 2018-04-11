# -*- coding: utf-8 -*-

import torch
import random
import argparse
from torch.utils.data import DataLoader
from driver.Config import Configurable
from driver.IO import *
from driver.DataLoader import *
from driver.Train import *


from torch.autograd import Variable

if __name__ == '__main__':
    # random
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    random.seed(666)
    np.random.seed(666)

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
    # dev_data = read_pkl(config.dev_pkl)
    test_data = read_pkl(config.test_pkl)

    feature_vec = read_pkl(config.load_feature_voc)
    label_vec = read_pkl(config.load_label_voc)

    if os.path.isfile(config.embedding_pkl):
        embedding = read_pkl(config.embedding_pkl)

    # DataLoader
    train_dataset = SentenceDataset(train_data, config.max_length, feature_vec, label_vec)
    # dev_dataset = SentenceDataset(dev_data, config.max_length, feature_vec, label_vec)
    test_dataset = SentenceDataset(test_data, config.max_length, feature_vec, label_vec)

    data_loader_train = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=args.thread)
    # data_loader_dev = DataLoader(
    #     dev_dataset, batch_size=config.batch_size, shuffle=False, num_workers=args.thread)
    data_loader_test = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=args.thread)

    # train
    train(data_loader_train, config, feature_vec, )
































