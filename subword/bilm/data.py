# !/usr/bin/env python3

import os
from glob import glob
import numpy as np
import random
import time

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
            return self._word_to_id[word]
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

def _get_batch(generator, batchsize, numsteps):
    cur_stream = [None]*batchsize
    no_more_data = False
    while True:
        inputs = np.zeros([batchsize, numsteps], np.int32)
        targets = np.zeros([batchsize, numsteps], np.int32)
        for i in range(batchsize):
            cur_pos = 0
            while cur_pos < numsteps:
                if cur_stream[i] is None or len(cur_stream[i]) <= 1:
                    try:
                        cur_stream[i] = list(next(generator))
                    except StopIteration:
                        no_more_data = True
                how_many = min(len(cur_stream[i])-1, numsteps-cur_pos)
                next_pos = cur_pos + how_many
                inputs[i, cur_pos:next_pos] = cur_stream[i][:how_many]
                targets[i, cur_pos:next_pos] = cur_stream[i][1:how_many+1]
                cur_pos = next_pos
                cur_stream[i] = cur_stream[i][how_many:]
        if no_more_data:
            break
        X = {'token_ids': inputs, 'target_ids': targets}
        yield X


class LMDataset(object):
    def __init__(self, filepattern, vocab, reverse=False, test=False,
                 shuffle_on_load=False):
        self._vocab = vocab
        self._all_shards = glob(filepattern)
        print('Found {} shards at {}'.format(len(self._all_shards), filepattern))
        self._shards_to_choose = []
        self._reverse = reverse
        self._test = test
        self._shuffle_on_load = shuffle_on_load
        self._ids = self._load_random_shard()

    def _choose_random_shards(self):
        if len(self._shards_to_choose) == 0:
            self._shards_to_choose = list(self._all_shards)
            random.shuffle(self._shards_to_choose)
        shard_name = self._shards_to_choose.pop()
        return shard_name

    def _load_shard(self, shard_name):
        print('Loading data from: {}'.format(shard_name))
        with open(shard_name, 'r', encoding='utf8') as f:
            sentences_raw = f.readlines()
        if self._reverse:
            sentences = []
            for sent in sentences_raw:
                splitted = sent.split()
                splitted.reverse()
                sentences.append(' '.join(splitted))
        else:
            sentences = sentences_raw

        if self._shuffle_on_load:
            random.shuffle(sentences)
        ids = [self.vocab.encode(sent, self._reverse) for sent in sentences]
        print('Loaded {} sentences.'.format(len(ids)))
        print('Finished loading!')
        return list(ids)


    def _load_random_shard(self):
        if self._test:
            if len(self._all_shards) == 0:
                raise StopIteration
            else:
                shard_name = self._all_shards.pop()
        else:
            shard_name = self._choose_random_shards()
        ids = self._load_shard(shard_name)
        self._i = 0
        self._nids = len(ids)
        return ids

    def get_sentence(self):
        while True:
            if self._i == self._nids:
                self._ids = self._load_random_shard()
            ret = self._ids[self._i]
            self._i += 1
            yield ret

    def iter_batches(self, batchsize, numsteps):
        for X in _get_batch(self.get_sentence(), batchsize, numsteps):
            yield X
            # return X

    @property
    def vocab(self):
        return self._vocab

class BidirectionalLMDataset(object):
    def __init__(self, filepattern, vocab, test=False, shuffle_on_load=False):
        self._forward = LMDataset(filepattern, vocab, reverse=False, test=test,
                                  shuffle_on_load=shuffle_on_load)
        self._backward = LMDataset(filepattern, vocab, reverse=True, test=test,
                                   shuffle_on_load=shuffle_on_load)
    def iter_batches(self, batchsize, numsteps):
        for X, Xr in zip(
            _get_batch(self._forward.get_sentence(), batchsize, numsteps),
            _get_batch(self._backward.get_sentence(), batchsize, numsteps)
            ):
            for k,v in Xr.items():
                X[k+'_reverse'] = v
            yield X



def test_LMDataset():
    vocab_file = '../data/example.vocab'
    vocab = Vocabulary(vocab_file)
    filepattern = '../data/*_seg_words.txt'
    ds = LMDataset(filepattern, vocab)
    data_batch = ds.iter_batches(4, 50)
    icnt = 0
    for idx, batch in enumerate(data_batch):
        print('inputs:\t' + vocab.decode(batch['token_ids'][0]))
        print('outputs:\t' + vocab.decode(batch['token_ids'][0]))
        # time.sleep(1)
        if icnt%10 == 0:
            break
        icnt += 1
    print('\n\n\n\n')
    print('===>when test mode:')
    ds = LMDataset(filepattern, vocab, test=True)
    data_batch = ds.iter_batches(512, 50)
    for idx, batch in enumerate(data_batch):
        print('inputs:\t' + vocab.decode(batch['token_ids'][0]))
        print('outputs:\t' + vocab.decode(batch['token_ids'][0]))
    print('\n\n\n\n')

def test_BidirectionalLMDataset():
    vocab_file = '../data/example.vocab'
    vocab = Vocabulary(vocab_file)
    filepattern = '../data/*_seg_words.txt'
    ds = BidirectionalLMDataset(filepattern, vocab)
    data_batch = ds.iter_batches(512, 50)
    for index, batch in enumerate(data_batch):
        print('inputs:\t' + vocab.decode(batch['token_ids'][0]))
        print('outputs:\t' + vocab.decode(batch['token_ids'][0]))
        print('inputs reverse:\t' + vocab.decode(batch['token_ids_reverse'][0]))
        print('outputs reverse:\t' + vocab.decode(batch['token_ids_reverse'][0]))
        print('\n')

if __name__ == '__main__':
    # test_LMDataset()
    test_BidirectionalLMDataset()
