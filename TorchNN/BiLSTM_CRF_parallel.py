# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def argmax1d(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp1d(vec):
    max_score = vec[0, argmax1d(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp2d(vec):
    label_size = vec.size()[0]

    max_score, max_ids = torch.max(vec, dim=0)
    max_score_broadcast = max_score.view(1, -1).expand(label_size, label_size)
    # t = torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=0))
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=0))


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp3d(vec, m_size):
    """
    calculate log of exp sum
    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
    return max_score.view(-1, m_size) + \
           torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)  # B * M


def log_sum_exp_song(scores, label_nums):
    """
    params:
        scores: variable (batch_size, label_nums, label_nums)
        label_nums
    return:
        variable (batch_size, label_nums)
    """
    batch_size = scores.size(0)
    max_scores, max_index = torch.max(scores, dim=1)
    ##### max_index: variable (batch_size, label_nums)
    ##### max_scores: variable (batch_size, label_nums)
    # max_scores = torch.gather(scores, 1, max_index.view(-1, 1, label_nums)).view(-1, 1, label_nums)
    max_score_broadcast = max_scores.unsqueeze(1).view(batch_size, 1, label_nums).expand(batch_size, label_nums, label_nums)
    return max_scores.view(batch_size, label_nums) + torch.log(torch.sum(torch.exp(scores - max_score_broadcast), 1)).view(batch_size, label_nums)



class CRFParallel(nn.Module):
    def __init__(self, config, num_embeddings, embedding_dim, padding_idx, label_size, embeddings):
        super(CRFParallel, self).__init__()
        self.config = config
        self.label_size = label_size

        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx)
        if embeddings is not None:
            self.embedding.from_pretrained(torch.from_numpy(embeddings))
        self.dropout = nn.Dropout(config.dropout_embed)
        self.lstm = nn.LSTM(embedding_dim, config.hidden_size, num_layers=config.num_layers, bidirectional=True)
        self.hidden2tag = nn.Linear(config.hidden_size * 2, label_size)

        self.START_TAG, self.STOP_TAG = label_size - 2, label_size - 1
        self.transitions = nn.Parameter(torch.randn(label_size, label_size), requires_grad=True)
        self.transitions.data[:, self.START_TAG] = -10000
        self.transitions.data[self.STOP_TAG, :] = -10000

    def _get_lstm_features(self, feats, lengths):
        """
        Args:
            feats: size=(seq_len, batch_size)
            lengths: list

        Returns:
            lstm_feats:
        """
        feats = self.embedding(feats)
        h = pack_padded_sequence(feats, lengths)
        h, _ = self.lstm(h)
        h, _ = pad_packed_sequence(h)
        h = self.dropout(h)

        lstm_feats = torch.squeeze(h, 1)
        lstm_feats = self.hidden2tag(lstm_feats)
        return lstm_feats

    def _forward_alg(self, feats, mask):
        """
        Args:
            feats: size=(seq_len, batch_size, label_size)
            mask: size=(batch_size, seq_len)

        Returns:
            score:
        """

        seq_len = feats.size(0)
        batch_size = feats.size(1)
        label_size = feats.size(2)
        assert label_size == self.label_size

        mask = torch.transpose(mask, 0, 1)
        ins_sum = seq_len * batch_size

        # 已经加好了转移得分
        feats = feats.view(ins_sum, 1, label_size).expand(ins_sum, label_size, label_size)
        scores = feats + self.transitions.view(1, label_size, label_size).expand(ins_sum, label_size, label_size)
        scores = scores.view(seq_len, batch_size, label_size, label_size)

        # 初始化第0个字
        seq_iter = enumerate(scores)
        _, inivalues = seq_iter.__next__()

        # 记录每一层
        partition = inivalues[:, self.START_TAG, :].clone().view(batch_size, label_size, 1)

        # 遍历每一层字
        for idx, cur_values in seq_iter:
            cur_values = cur_values + \
                         partition.view(batch_size, label_size, 1).expand(batch_size, label_size, label_size)
            cur_partition = log_sum_exp3d(cur_values, label_size)

            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, label_size)

            masked_cur_partition = cur_partition.masked_select(mask_idx)
            if masked_cur_partition.dim() != 0:
                mask_idx = mask_idx.view(batch_size, label_size, 1)
                partition.masked_scatter_(mask_idx, masked_cur_partition)

        cur_values = self.transitions.view(1, label_size, label_size).expand(batch_size, label_size, label_size) + \
                     partition.view(batch_size, label_size, 1).expand(batch_size, label_size, label_size)
        cur_partition = log_sum_exp3d(cur_values, label_size)
        final_partition = cur_partition[:, self.STOP_TAG]
        return final_partition.sum(), scores

    def _score_sentence(self, scores, tags, mask):
        """
        Args:
            scores: size=(seq_len, batch_size, label_size, label_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)

        Returns:
            score:
        """
        seq_len = scores.size(0)
        batch_size = scores.size(1)
        label_size = scores.size(2)
        new_tags = torch.zeros((batch_size, seq_len), dtype=torch.int64)
        if self.config.use_cuda:
            new_tags = new_tags.cuda()
        for idx in range(seq_len):
            if idx == 0:
                new_tags[:, 0] = (label_size - 2) * label_size + tags[:, 0]

            else:
                new_tags[:, idx] = tags[:, idx - 1] * label_size + tags[:, idx]

        end_transition = self.transitions[:, self.STOP_TAG].view(1, label_size).expand(batch_size, label_size)
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        end_ids = torch.gather(tags, 1, length_mask - 1)
        end_energy = torch.gather(end_transition, 1, end_ids)

        new_tags = new_tags.transpose(1, 0).view(seq_len, batch_size, 1)
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))

        # ## calculate the score from START_TAG to first label
        # start_transition = self.transitions[START_TAG,:].view(1, tag_size).expand(batch_size, tag_size)
        # start_energy = torch.gather(start_transition, 1, tags[0,:])

        ## add all score together
        # gold_score = start_energy.sum() + tg_energy.sum() + end_energy.sum()
        gold_score = tg_energy.sum() + end_energy.sum()
        return gold_score

    def _viterbi_decode(self, feats, mask):
        """
            input:
                feats: (seq_len, batch, label_size)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, seq_len) decoded sequence
                path_score: (batch, 1) corresponding score for each sequence (to be implementated)
        """
        seq_len = feats.size(0)
        batch_size = feats.size(1)
        label_size = feats.size(2)
        assert label_size == self.label_size

        # mask
        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()
        mask = mask.transpose(0, 1)
        ins_sum = seq_len * batch_size
        # score
        feats = feats.view(ins_sum, 1, label_size).expand(ins_sum, label_size, label_size)
        scores = feats + self.transitions.view(1, label_size, label_size).expand(ins_sum, label_size, label_size)
        scores = scores.view(seq_len, batch_size, label_size, label_size)

        # 遍历
        seq_iter = enumerate(scores)
        back_points = []
        partition_history = []

        mask = (1 - mask.long()).byte()

        _, inivalues = seq_iter.__next__()
        partition = inivalues[:, self.START_TAG, :].clone().view(batch_size, label_size)
        partition_history.append(partition)

        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.view(batch_size, label_size, 1).expand(batch_size, label_size, label_size)
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition)
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, label_size), 0)
            back_points.append(cur_bp)

        partition_history = torch.cat(partition_history, 0).view(seq_len, batch_size, -1).transpose(1, 0)
        last_position = length_mask.view(batch_size, 1, 1).expand(batch_size, 1, label_size) - 1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size, label_size, 1)

        last_values = last_partition.expand(batch_size, label_size, label_size) + \
                      self.transitions.view(1,label_size, label_size).expand(batch_size, label_size, label_size)
        _, last_bp = torch.max(last_values, 1)
        pad_zero = torch.zeros(batch_size, label_size).long()
        if self.config.use_cuda:
            pad_zero = pad_zero.cuda()
        back_points.append(pad_zero)
        back_points  =  torch.cat(back_points).view(seq_len, batch_size, label_size)

        ## select end ids in STOP_TAG
        pointer = last_bp[:, self.STOP_TAG]
        insert_last = pointer.view(batch_size,1,1).expand(batch_size,1, label_size)
        back_points = back_points.transpose(1,0)
        ## move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values
        # print "lp:",last_position
        # print "il:",insert_last
        back_points.scatter_(1, last_position, insert_last)
        # print "bp:",back_points
        # exit(0)
        back_points = back_points.transpose(1,0)
        ## decode from the end, padded position ids are 0, which will be filtered if following evaluation
        decode_idx = torch.LongTensor(seq_len, batch_size)
        if self.config.use_cuda:
            decode_idx = decode_idx.cuda()
        decode_idx[-1] = pointer.detach()
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.view(batch_size, 1))
            decode_idx[idx] = pointer.detach().view(batch_size)
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        return path_score, decode_idx

    def neg_log_likelihood(self, feats, tags, lengths, mask):
        """
        Args:
            feats: size=(seq_len, batch_size)
            tags: size=(batch_size, seq_len)
            lengths: list
            mask: size=(batch_size, seq_len)
        """
        lstm_feats = self._get_lstm_features(feats, lengths)

        forward_score, scores = self._forward_alg(lstm_feats, mask)
        gold_score = self._score_sentence(scores, tags, mask)

        return forward_score - gold_score

    def forward(self, feats, lengths, mask):
        lstm_feats = self._get_lstm_features(feats, lengths)
        score, tag_seq = self._viterbi_decode(lstm_feats, mask)
        return score, tag_seq
