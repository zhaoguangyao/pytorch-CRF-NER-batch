# -*- coding: utf-8 -*-

import os
import torch
import random
import argparse
import numpy as np
from torch.utils.data import DataLoader
from driver.Config import Configurable
from driver.IO import *
from driver.Split import *


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

    # DataLoader

































