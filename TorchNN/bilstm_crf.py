#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    Sequence Labeling Model
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from . import CRF


class BiLSTMCRFModel(nn.Module):

    def __init__(self, args):
        """
        Args:
            feature_size_dict: dict, 特征表大小字典
            feature_dim_dict: dict, 输入特征dim字典
            pretrained_embed: np.array, default is None
            dropout_rate: float, dropout rate
            use_cuda: bool
        """
        super(BiLSTMCRFModel, self).__init__()
        for k, v in args.items():
            self.__setattr__(k, v)

        # feature embedding layer
        self.embedding_dict = dict()
        lstm_input_dim = 0
        for feature_name in self.features:
            embed = nn.Embedding(
                self.feature_size_dict[feature_name], self.feature_dim_dict[feature_name])
            self.embedding_dict[feature_name] = embed
            if feature_name != 'label':
                lstm_input_dim += self.feature_dim_dict[feature_name]
        if hasattr(self, 'pretrained_embed') and self.pretrained_embed is not None:
            self.embedding_dict[self.features[0]].weight.data.copy_(torch.from_numpy(self.pretrained_embed))
            if self.use_cuda:
                self.embedding_dict[self.features[0]].weight.data = \
                    self.embedding_dict[self.features[0]].weight.data.cuda()

        # lstm layer
        self.lstm = nn.LSTM(lstm_input_dim, self.lstm_units, num_layers=self.layer_nums, bidirectional=True)

        # init crf layer
        self.target_size = self.feature_size_dict['label']
        args_crf = {'target_size': self.target_size, 'use_cuda': self.use_cuda}
        self.crf = CRF(args_crf)

        # dropout layer
        if not hasattr(self, 'dropout_rate'):
            self.__setattr__('dropout_rate', '0.5')
        self.dropout = nn.Dropout(self.dropout_rate)

        # dense layer
        self.hidden2tag = nn.Linear(self.lstm_units*2, self.target_size+2)

        self._init_weight()

    def get_lstm_feature(self, input_dict, batch_size):
        """
        Returns:
            tag_scores: size=[batch_size * max_len, nb_classes]
        """
        # concat inputs
        inputs = []
        for feature_name in self.features:
            inputs.append(self.embedding_dict[feature_name](input_dict[feature_name]))
        inputs = torch.cat(inputs, dim=2)  # size=[batch_size, max_len, input_size]

        inputs = torch.transpose(inputs, 1, 0)  # size=[max_len, batch_size, input_size]

        self.lstm.flatten_parameters()
        lstm_output, _ = self.lstm(inputs)
        lstm_output = lstm_output.transpose(1, 0).contiguous()  # [batch_size, max_len, lstm_units]

        # [batch_size * max_len, target_size]
        lstm_feats = self.hidden2tag(lstm_output.view(-1, self.lstm_units*2))

        return lstm_feats.view(batch_size, self.max_len, -1)

    def loss(self, feats, mask, tags):
        return self.crf.neg_log_likelihood_loss(feats, mask, tags)

    def forward(self, input_dict):
        """
        Args:
            inputs: autograd.Variable, size=[batch_size, max_len]
        """
        batch_size = input_dict[self.features[0]].size()[0]

        return self.get_lstm_feature(input_dict, batch_size)

    def predict(self, input_dict):
        """
        预测标签
        """
        # viterbi decode: self.crf(...)
        batch_size = input_dict[self.features[0]].size()[0]
        lstm_feats = self.get_lstm_feature(input_dict, batch_size)
        lstm_feats = lstm_feats.view(batch_size, self.max_len, -1)  # [batch_size, max_len, -1]
        word_inputs = input_dict[self.features[0]]
        mask = word_inputs > 0
        path_score, best_paths = self.crf(lstm_feats, mask)
        actual_lens = torch.sum(word_inputs>0, dim=1).int()

        tags_list = []
        for i in range(batch_size):
            tags_list.append(best_paths[i].cpu().data.numpy()[:actual_lens.data[i]])

        return tags_list

    def _init_weight(self, scope=.1):
        if hasattr(self, 'pretrained_embed') and self.pretrained_embed is None:
            self.embedding_dict[self.features[0]].weight.data.uniform_(-scope, scope)
            if self.use_cuda:
                self.embedding_dict[self.features[0]].weight.data = \
                    self.embedding_dict[self.features[0]].weight.data.cuda()
        for feature_name in self.features[1:]:
            self.embedding_dict[feature_name].weight.data.uniform_(-scope, scope)
            if self.use_cuda:
                self.embedding_dict[feature_name].weight.data = \
                    self.embedding_dict[feature_name].weight.data.cuda()
        self.hidden2tag.weight.data.uniform_(-scope, scope)

    def set_use_cuda(self, use_cuda):
        self.use_cuda = use_cuda
