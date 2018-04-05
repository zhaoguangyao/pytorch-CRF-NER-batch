# -*- coding: utf-8 -*-

from configparser import ConfigParser
import os


class Configurable(object):
    def __init__(self, config_file, extra_args):
        config = ConfigParser()
        config.read(config_file)
        if extra_args:
            extra_args = dict([(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)

        self._config = config
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        config.write(open(self.config_file, 'w'))
        print('Load config file successfully.\n')
        for section in config.sections():
            for k, v in config.items(section):
                print(k, v)

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
    def shuffle(self):
        return self._config.getboolean('Data', 'shuffle')

    @property
    def vocab_size(self):
        return self._config.getint('Data', 'vocab_size')

    @property
    def save_dir(self):
        return self._config.get('Save', 'save_dir')

    @property
    def config_file(self):
        return self._config.get('Save', 'config_file')

    @property
    def save_model_path(self):
        return self._config.get('Save', 'save_model_path')

    @property
    def save_src_vocab_path(self):
        return self._config.get('Save', 'save_src_vocab_path')

    @property
    def save_tgt_vocab_path(self):
        return self._config.get('Save', 'save_tgt_vocab_path')

    @property
    def load_dir(self):
        return self._config.get('Save', 'load_dir')

    @property
    def load_model_path(self):
        return self._config.get('Save', 'load_model_path')

    @property
    def load_src_vocab_path(self):
        return self._config.get('Save', 'load_src_vocab_path')

    @property
    def load_tgt_vocab_path(self):
        return self._config.get('Save', 'load_tgt_vocab_path')

    @property
    def embedding_file(self):
        return self._config.get('Network', 'embedding_file')

    @property
    def embed_dim(self):
        return self._config.getint('Network', 'embed_dim')

    @property
    def hidden_size(self):
        return self._config.getint('Network', 'hidden_size')

    @property
    def attention_size(self):
        return self._config.getint('Network', 'attention_size')

    @property
    def dropout_embed(self):
        return self._config.getfloat('Network', 'dropout_embed')

    @property
    def dropout_rnn(self):
        return self._config.getfloat('Network', 'dropout_rnn')

    @property
    def max_norm(self):
        return self._config.getfloat('Network', 'max_norm')

    @property
    def which_model(self):
        return self._config.get('Network', 'which_model')

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

    @property
    def epochs(self):
        return self._config.getint('Run', 'epochs')

    @property
    def batch_size(self):
        return self._config.getint('Run', 'batch_size')

    @property
    def log_interval(self):
        return self._config.getint('Run', 'log_interval')

    @property
    def test_interval(self):
        return self._config.getint('Run', 'test_interval')

    @property
    def save_after(self):
        return self._config.getint('Run', 'save_after')




