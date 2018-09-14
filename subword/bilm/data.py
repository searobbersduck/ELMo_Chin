# !/usr/bin/env python3

import os
from glob import glob
import numpy as np

class Vocabulary(object):
    def __init__(self, filename, vadidate_file=False):
        self._id_to_word = []
        self._word_to_id = {}
        self._unk = -1
        self._bos = -1
        self._eos = -1
        idx = 0
        with open(filename, 'r', encoding='utf8') as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or line == '':
                    continue
                ss = line.split('\t')
                if len(ss) != 2:
                    continue
                if ss[0] == '<s>':
                    self._bos = idx
                elif ss[0] == '</s>':
                    self._eos = idx
                elif ss[0] == '<unk>':
                    self._unk = idx
                self._id_to_word.append(ss[0])
                self._word_to_id[ss[0]] = idx
                idx += 1
        self._size = len(self._id_to_word)
        if vadidate_file:
            if self._bos == -1 or self._eos == -1 or self._unk == -1:
                raise ValueError("Ensure the vocabulary file has"
                                 "<s>, </s>, <unk> tokens!")
    @property
    def bos(self):
        return self._bos

    @property
    def eos(self):
        return self._eos

    @property
    def unk(self):
        return self._unk

    @property
    def size(self):
        return len(self._id_to_word)

    def word_to_id(self, word):
        if word in self._word_to_id:
            return self._word_to_id['word']
        return self._unk

    def id_to_word(self, id):
        if id >= self._size:
            return '<unk>'
        return self._id_to_word[id]

    def decode(self, ids):
        return ' '.join([self.id_to_word(id) for id in ids])

    def encode(self, sentence, reverse=False):
        # todo: need to check
        ids = [self.word_to_id(word) for word in sentence.split()]
        if reverse:
            return np.array([self.eos] + ids + [self.bos], dtype=np.int32)
        else:
            return np.array([self.bos] + ids + [self.eos], dtype=np.int32)

class LMDataset(object):
