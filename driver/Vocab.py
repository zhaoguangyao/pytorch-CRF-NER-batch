# -*- coding: utf-8 -*-
import numpy as np

PAD, UNK = 0, 1
PAD_S, UNK_S = '<pad>', '<unk>'


class VocabSrc:
    def __init__(self, word_list):
        # no fine tune
        self._id2extword = [PAD_S, UNK_S]

        self.i2w = [PAD_S, UNK_S] + word_list
        self.w2i = {}
        for idx, word in enumerate(self.i2w):
            self.w2i[word] = idx
        if len(self.w2i) != len(self.i2w):
            print("serious bug: words dumplicated, please check!")

    def word2id(self, xx):
        if isinstance(xx, list):
            return [self.w2i.get(word, UNK) for word in xx]
        return self.w2i.get(xx, UNK)

    def id2word(self, xx):
        if isinstance(xx, list):
            return [self.i2w[idx] for idx in xx]
        return self.i2w[xx]

    @property
    def size(self):
        return len(self.i2w)

    @property
    def embed_size(self):
        return len(self._id2extword)

    # 这个是extendEmbedding
    def create_extend_embs(self, embfile):
        embedding_dim = -1
        word_count = 0
        with open(embfile, encoding='utf-8') as f:
            for line in f:
                if word_count < 1:
                    values = line.split(' ')
                    embedding_dim = len(values) - 1
                word_count += 1
        print('\nTotal words: ' + str(word_count))
        print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')

        index = len(self._id2extword)
        embeddings = np.zeros((word_count + index, embedding_dim))
        with open(embfile, encoding='utf-8') as f:
            for line in f:
                values = line.split(' ')
                self._id2extword.append(values[0])
                vector = np.array(values[1:], dtype='float64')
                embeddings[UNK] += vector
                embeddings[index] = vector
                index += 1

        embeddings[UNK] = embeddings[UNK] / word_count
        embeddings = embeddings / np.std(embeddings)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._extword2id = reverse(self._id2extword)

        if len(self._extword2id) != len(self._id2extword):
            print("serious bug: extern words dumplicated, please check!")

        return embeddings

    # fine tune embs
    def create_vocab_embs(self, embed_file):
        first_line = False
        embed_dim = -1
        embed_word_count = 0
        with open(embed_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                strs = line.split(' ')
                if len(strs) == 2:
                    embed_word_count = int(strs[0])
                    embed_dim = int(strs[1])
                    first_line = True
                    print('---')
                    break
                if embed_word_count < 1:
                    values = line.split()
                    embed_dim = len(values) - 1
                embed_word_count += 1
        print('\nTotal words: ' + str(embed_word_count))
        print('The dim of pretrained embeddings: ' + str(embed_dim) + '\n')

        find_count = 0
        embeddings = np.zeros((len(self.i2w), embed_dim))
        ii = 0
        with open(embed_file, 'r', encoding='utf-8') as f:
            if first_line:
                for line in f:
                    if ii > 0:
                        line = line.strip()
                        values = line.split(' ')
                        if values[0] in self.w2i:
                            vector = np.array(values[1:], dtype='float64')
                            embeddings[self.w2i[values[0]]] = vector
                            embeddings[UNK] += vector
                            find_count += 1
                    ii += 1
                    # print(ii)
            else:
                for line in f:
                    line = line.strip()
                    values = line.split(' ')
                    if values[0] in self.w2i:
                        vector = np.array(values[1:], dtype='float64')
                        embeddings[self.w2i[values[0]]] = vector
                        embeddings[UNK] += vector
                        find_count += 1

        print("The number of vocab word find in extend embedding is: ", str(find_count))
        print("The number of all vocab is: ", str(len(self.w2i)))
        embeddings[UNK] = embeddings[UNK] / find_count
        embeddings = embeddings / np.std(embeddings)

        not_find = len(self.w2i) - find_count
        oov_ratio = float(not_find / len(self.w2i))

        print('oov ratio: {:.4f}'.format(oov_ratio))

        return embeddings, embed_dim


class VocabTgt:
    def __init__(self, word_list):
        self.i2w = word_list
        self.w2i = {}
        for idx, word in enumerate(self.i2w):
            self.w2i[word] = idx
        if len(self.w2i) != len(self.i2w):
            print("serious bug: words dumplicated, please check!")

    def word2id(self, xx):
        if isinstance(xx, list):
            return [self.w2i.get(word) for word in xx]
        return self.w2i.get(xx)

    def id2word(self, xx):
        if isinstance(xx, list):
            return [self.i2w[idx] for idx in xx]
        return self.i2w[xx]

    @property
    def size(self):
        return len(self.i2w)


def add_extend_word(new_words, origin_words):
    for word in new_words:
        if word in origin_words:
            continue
        else:
            origin_words.append(word)
    return origin_words
