# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BILSTM(nn.Module):
    def __init__(self, config, embed_size, embed_dim, padding_id, embedding=None):
        super(BILSTM, self).__init__()

        self.embed = nn.Embedding(embed_size, embed_dim, padding_idx=padding_id)
        if embedding is not None:
            self.embed.weight.data.copy_(torch.from_numpy(embedding))
            self.dropout = nn.Dropout(config.dropout_embed)

        self.bilstm = nn.LSTM(embed_dim, config.hidden_size, dropout=config.dropout_rnn, bidirectional=True)

    def forward(self, x, x_lengths):
        x = self.embed(x)
        x = self.dropout(x)

        x = pack_padded_sequence(x, x_lengths)
        x, _ = self.bilstm(x)
        x, _ = pad_packed_sequence(x)
        return x


