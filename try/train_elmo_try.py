import sys
# sys.path.append('../')
import argparse

import numpy as np

from bilm.training import train_with_single_core, load_vocab
from bilm.data import BidirectionalLMDataset

def main(args):
    print('h0')
    vocab = load_vocab(args.vocab_file, 10)
    print('h1')
    batch_size = 64
    n_gpus = 3
    n_train_tokens = 4775300
    options = {
        'bidirectional': True,

        'char_cnn': {'activation': 'relu',
                     'embedding': {'dim': 16},
                     'filters': [[1, 32],
                                 [2, 32],
                                 [3, 64],
                                 [4, 128],
                                 [5, 256],
                                 [6, 512],
                                 [7, 1024]],
                     'max_characters_per_token': 10,
                     'n_characters': 105047,
                     'n_highway': 2},

        'dropout': 0.1,

        'lstm': {
            'cell_clip': 3,
            'dim': 4096,
            'n_layers': 2,
            'proj_clip': 3,
            'projection_dim': 512,
            'use_skip_connections': True},

        'all_clip_norm_val': 10.0,

        'n_epochs': 10,
        'n_train_tokens': n_train_tokens,
        'batch_size': batch_size,
        'n_tokens_vocab': vocab.size,
        'unroll_steps': 20,
        'n_negative_samples_batch': 8192,
    }
    inpattern = args.train_prefix
    data = BidirectionalLMDataset(inpattern, vocab, test=False,
                                  shuffle_on_load=True)
    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    train_with_single_core(options, data, tf_save_dir, tf_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_prefix', help='Prefix for train files')

    args = parser.parse_args()
    main(args)