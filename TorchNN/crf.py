# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from torch.nn import Parameter
from torch.autograd import Variable
from .lstm import BILSTM


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class CRF(nn.Module):
    # def __init__(self, config, embed_size, embed_dim, label_voc, padding_id, start,
    #              stop, embedding=None):
    def __init__(self, config, embed_size, embed_dim, label_voc, padding_id, embedding=None):
        super(CRF, self).__init__()
        self.config = config
        # self.start = start
        # self.stop = stop
        self.label_voc = label_voc

        self.bilstm = BILSTM(config, embed_size, embed_dim, padding_id, embedding)

        self.hidden2tag = nn.Linear(config.hidden_size * 2, label_voc.size)

        # self.START_TAG_IDX, self.END_TAG_IDX = start, stop
        self.transfer_score = Parameter(torch.randn(label_voc.size, label_voc.size))
        init.xavier_normal(self.transfer_score)

    def _froward_alg(self, feats, input_length):
        # Do the forward algorithm to compute the partition function
        batch_size = len(input_length)
        max_length = input_length[0]

        mask_list = []
        for idx, v in enumerate(input_length):
            m_list = []
            for idy in range(max_length):
                if idy <= v:
                    m_list.append(1)
                else:
                    m_list.append(0)
            mask_list.append(m_list)
        mask_np = np.array(mask_list)
        mask = Variable(torch.from_numpy(mask_np).type(torch.FloatTensor))
        if self.config.use_cuda:
            mask = mask.cuda()

        # 只是因为目前batch_size = 1
        feats = torch.squeeze(feats, 1)

        # feats = torch.transpose(feats, 0, 1).contiguous()

        # batch个得分
        # init_alphas = Variable(torch.FloatTensor(batch_size).fill_(0))
        # 保留每一个位置上的得分
        scores = Variable(torch.zeros(max_length, self.label_voc.size))
        scores[0] = feats[0]
        feats = feats.unsqueeze(2).expand(feats.size(0), feats.size(1), feats.size(1))
        t = feats[0][0]
        for idx in range(1, max_length):
            for idy in range(self.label_voc.size):
                init_alphas = scores[idx - 1] + feats[idx, idy] + self.transfer_score[:, idy]
                scores[idx, idy] = log_sum_exp(init_alphas.unsqueeze(0))

        t = scores[-1].unsqueeze(0)
        sentence_score = log_sum_exp(t)
        return sentence_score

    def _score_gold(self, feats, input_length, target):
        # 先写batch = 1的，也就是targets的第一维=1
        max_length = input_length[0]

        # 只是因为目前batch_size = 1
        feats = torch.squeeze(feats, 1)
        target = torch.squeeze(target, 1)

        score = Variable(torch.zeros(1))
        score += feats[0, target.data[0]]
        pre_trans = target.data[0]
        for idx in range(1, max_length):
            score += feats[idx, target.data[idx]]
            score += self.transfer_score[pre_trans, target.data[idx]]
            pre_trans = target.data[idx]
        return score

    def get_loss(self, h, input_length, target):
        sentence_score = self._froward_alg(h, input_length)
        gold_score = self._score_gold(h, input_length, target)
        loss = sentence_score - gold_score
        return loss

    def forward(self, x, input_length):
        h = self.bilstm(x, input_length)
        # 过这个线性层会导致原本为0的padding的地方 不为 0了
        h = self.hidden2tag(h)
        return h

    def _viterbi_decode(self, feats, input_length):
        batch_size = len(input_length)
        max_length = input_length[0]

        mask_list = []
        for idx, v in enumerate(input_length):
            m_list = []
            for idy in range(max_length):
                if idy <= v:
                    m_list.append(1)
                else:
                    m_list.append(0)
            mask_list.append(m_list)
        mask_np = np.array(mask_list)
        mask = Variable(torch.from_numpy(mask_np).type(torch.FloatTensor))
        if self.config.use_cuda:
            mask = mask.cuda()

        # 只是因为目前batch_size = 1
        feats = torch.squeeze(feats, 1)

        scores = Variable(torch.zeros(max_length, self.label_voc.size))
        scores[0] = feats[0]
        feats = feats.unsqueeze(2).expand(feats.size(0), feats.size(1), feats.size(1))
        path = np.zeros((max_length, self.label_voc.size)) - 1
        for idx in range(1, max_length):
            for idy in range(self.label_voc.size):
                init_alphas = scores[idx - 1] + feats[idx, idy] + self.transfer_score[:, idy]
                m_max = torch.max(init_alphas, 0)
                scores[idx, idy] = m_max[0]
                path[idx][idy] = int(m_max[1].data[0])
        init_alphas = scores[max_length - 1]
        m_max = torch.max(init_alphas, 0)
        final_path = [int(m_max[1].data[0])]
        count = max_length - 1
        pos = int(m_max[1].data[0])
        while count >= 1:
            final_path.append(int(path[count][pos]))
            pos = int(path[count][pos])
            count -= 1
        return list(reversed(final_path))
