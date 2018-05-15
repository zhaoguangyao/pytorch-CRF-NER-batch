# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import Parameter

class CRF(nn.Module):
    def __init__(self):
        super(CRF, self).__init__()

        self.transfer_score = Parameter()

    def forward(self, input, input_length):
        pass

    def _score_sentence(self, feats, mask):
        transfer_score = P

    def _viterbi_decode(self, feats, mask):
        pass