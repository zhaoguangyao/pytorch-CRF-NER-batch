# -*- coding: utf-8 -*-
import os
from configparser import ConfigParser


class Configurable(object):
    def __init__(self, config_file, extra_args):
        config = ConfigParser()
        config.read(config_file)
        if extra_args:
            extra_args = dict([(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
        else:
            extra_args = dict()
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)

        self._config = config
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
        config.write(open(self.config_file, 'w'))
        print('Load config file successfully.\n')
        for section in config.sections():
            for k, v in config.items(section):
                print(k, v)

    # Network
    @property
    def embed_dim(self):
        return self._config.getint('Network', 'embed_dim')

    @property
    def num_layers(self):
        return self._config.getint('Network', 'num_layers')

    @property
    def hidden_size(self):
        return self._config.getint('Network', 'hidden_size')

    @property
    def dropout_embed(self):
        return self._config.getfloat('Network', 'dropout_embed')

    @property
    def dropout_rnn(self):
        return self._config.getfloat('Network', 'dropout_rnn')

    @property
    def max_norm(self):
        return self._config.getfloat('Network', 'max_norm')

    # Data
    @property
    def data_dir(self):
        return self._config.get('Data', 'data_dir')

    @property
    def train_file(self):
        return self._config.get('Data', 'train_file')

    @property
    def dev_file(self):
        return self._config.get('Data', 'dev_file')

    @property
    def test_file(self):
        return self._config.get('Data', 'test_file')

    @property
    def embedding_file(self):
        return self._config.get('Data', 'embedding_file')

    @property
    def vocab_size(self):
        return self._config.getint('Data', 'vocab_size')

    @property
    def max_length(self):
        return self._config.getint('Data', 'max_length')

    @property
    def shuffle(self):
        return self._config.getboolean('Data', 'shuffle')

    # SaveAndLoad
    @property
    def dir(self):
        return self._config.get('SaveAndLoad', 'dir')

    @property
    def model_path(self):
        return self._config.get('SaveAndLoad', 'model_path')

    @property
    def feature_voc(self):
        return self._config.get('SaveAndLoad', 'feature_voc')

    @property
    def label_voc(self):
        return self._config.get('SaveAndLoad', 'label_voc')

    @property
    def config_file(self):
        return self._config.get('SaveAndLoad', 'config_file')

    @property
    def train_pkl(self):
        return self._config.get('SaveAndLoad', 'train_pkl')

    @property
    def dev_pkl(self):
        return self._config.get('SaveAndLoad', 'dev_pkl')

    @property
    def test_pkl(self):
        return self._config.get('SaveAndLoad', 'test_pkl')

    @property
    def embedding_pkl(self):
        return self._config.get('SaveAndLoad', 'embedding_pkl')

    # Optimizer
    @property
    def learning_algorithm(self):
        return self._config.get('Optimizer', 'learning_algorithm')

    @property
    def lr(self):
        return self._config.getfloat('Optimizer', 'lr')

    @property
    def lr_scheduler(self):
        return self._config.get('Optimizer', 'lr_scheduler')

    @property
    def weight_decay(self):
        return self._config.getfloat('Optimizer', 'weight_decay')

    @property
    def clip_norm(self):
        return self._config.getfloat('Optimizer', 'clip_norm')

    # Run
    @property
    def epochs(self):
        return self._config.getint('Run', 'epochs')

    @property
    def batch_size(self):
        return self._config.getint('Run', 'batch_size')

    @property
    def test_interval(self):
        return self._config.getint('Run', 'test_interval')

    @property
    def save_after(self):
        return self._config.getint('Run', 'save_after')
