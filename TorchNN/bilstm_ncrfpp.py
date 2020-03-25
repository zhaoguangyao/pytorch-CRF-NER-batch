import torch
import torch.nn as nn
from .ncrfpp import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTM_NCRFPP(nn.Module):
    def __init__(self, config, num_embeddings, embedding_dim, padding_idx, label_size, embeddings):
        super(BiLSTM_NCRFPP, self).__init__()
        self.config = config
        self.label_size = label_size

        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx)
        if embeddings is not None:
            self.embedding.from_pretrained(torch.from_numpy(embeddings))
        self.dropout = nn.Dropout(config.dropout_embed)
        self.lstm = nn.LSTM(embedding_dim, config.hidden_size, num_layers=config.num_layers, bidirectional=True)
        self.hidden2tag = nn.Linear(config.hidden_size * 2, label_size)

        self.ncrfpp = CRF(target_size=label_size - 2, config=config)

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

    def neg_log_likelihood(self, feats, tags, lengths, mask):
        """
        Args:
            feats: size=(seq_len, batch_size)
            tags: size=(batch_size, seq_len)
            lengths: list
            mask: size=(batch_size, seq_len)
        """
        lstm_feats = self._get_lstm_features(feats, lengths)
        lstm_feats = torch.transpose(lstm_feats, 0, 1)
        loss = self.ncrfpp.neg_log_likelihood_loss(lstm_feats, mask, tags)
        return loss

    def forward(self, feats, lengths, mask):
        lstm_feats = self._get_lstm_features(feats, lengths)
        lstm_feats = torch.transpose(lstm_feats, 0, 1)
        score, tag_seq = self.ncrfpp.viterbi_decode(lstm_feats, mask)
        return score, tag_seq

