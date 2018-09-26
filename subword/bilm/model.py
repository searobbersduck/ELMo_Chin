# !/usr/bin/env python3

import sys
import os
import numpy as np
from glob import glob
import tensorflow as tf

class LanguageModel(object):
    def __init__(self, options, is_training):
        self.options = options
        self.is_training = is_training
        self.bidirectional = self.options.get('bidirectional', False)
        self.share_embedding_softmax = self.options.get('share_embedding_softmax', False)
        self.sample_softmax = self.options.get('sample_softmax', True)
        self._build()

    def _build_word_embeddings(self):
        batch_size = self.options['batch_size']
        unroll_steps = self.options['unroll_steps']
        n_tokens_vocab = self.options['n_tokens_vocab']
        projection_dim = self.options['lstm']['projection_dim']
        self.token_ids = tf.placeholder(tf.int64, shape=(batch_size, unroll_steps), name='token_ids')
        with tf.device('/cpu:0'):
            self.embedding_weights = self.get_varaible('embedding_weights',
                                                       [n_tokens_vocab, projection_dim],
                                                       dtype=tf.float32)
            self.embedding = tf.nn.embedding_lookup(self.embedding_weights, self.token_ids)
        if self.bidirectional:
            self.token_ids_reverse = tf.placeholder(tf.int64, shape=(batch_size, unroll_steps), name='token_ids_reverse')
            with tf.device('/cpu:0'):
                self.embedding_reverse = tf.nn.embedding_lookup(self.embedding_weights, self.token_ids_reverse)

    def _build(self):
        n_tokens_vocab = self.options['n_tokens_vocab']
        batch_size = self.options['batch_size']
        unroll_steps = self.options['unroll_steps']
        projection_dim = self.options['lstm']['projection_dim']
        lstm_dim = self.options['lstm']['lstm_dim']
        dropout = self.option['dropout']
        keep_prob = 1.0 - dropout
        n_lstm_layers = self.options['lstm']['n_lstm_layers']
        self._build_word_embeddings()
        lstm_outputs =[]
        if self.bidirectional:
            lstm_inputs = [self.embedding, self.embedding_reverse]
        else:
            lstm_inputs = [self.embedding]
        use_skip_connections = tf.options['use_skip_connections']
        if use_skip_connections:
            print('use skip connections!')
        cell_clip = self.cell_clip
        proj_clip = self.proj_clip
        self.init_lstm_state = []
        self.final_lstm_state = []
        for index, lstm_input in enumerate(lstm_inputs):
            lstm_cells = []
            for i in range(n_lstm_layers):
                if projection_dim > lstm_dim:
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_dim,
                                                        cell_clip=cell_clip,
                                                        proj_clip=proj_clip)
                else:
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_dim, projection_dim,
                                                        cell_clip=cell_clip,
                                                        proj_clip=proj_clip)
                if use_skip_connections:
                    if i == 0:
                        pass
                    else:
                        lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)
                if self.is_training:
                    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=keep_prob)
                lstm_cells.append(lstm_cell)
            if n_lstm_layers > 1:
                lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
            else:
                lstm_cell = lstm_cells[0]
            with tf.control_dependecies([lstm_input]):
                self.init_lstm_state.append(lstm_cell.zero_state(batch_size, tf.float32))
                if self.bidirectional:
                    with tf.variable_scope('RNN_{}'.format(index)):
                        _lstm_output_unpacked, final_state = tf.nn.static_rnn(
                            lstm_cell, tf.unstack(lstm_input, axis=1),
                            initial_state = self.init_lstm_state[-1]
                        )
                else:
                    _lstm_output_unpacked, final_state = tf.nn.static_rnn(
                        lstm_cell, tf.unstatck(lstm_input, axis=1),
                        initial_state=self.init_lstm_state[-1]
                    )
            lstm_output_flat = tf.reshape(
                tf.stack(_lstm_output_unpacked, axis=1),
                [-1, projection_dim]
            )
            if self.is_training:
                lstm_output_flat = tf.nn.dropout(lstm_output_flat, keep_prob)
            tf.add_to_collection('lstm_output_embeddings', _lstm_output_unpacked)
            lstm_outputs.append(lstm_output_flat)
        self._build_loss(lstm_outputs)

    def _build_loss(self, lstm_outputs):
        n_tokens_vocab = self.options['n_tokens_vocab']
        batch_size = self.options['batch_size']
        unroll_steps = self.options['unroll_steps']
        def _get_next_token_placeholder(suffix):
            name = 'next_token_ids' + suffix
            id_placeholder = tf.placeholder(tf.int64, shape=(batch_size, unroll_steps),
                                            name=name)
            return id_placeholder
        self.next_token_ids = _get_next_token_placeholder('')
        if self.bidirectional:
            self.next_token_ids_reverse = _get_next_token_placeholder('_reverse')
        softmax_dim = self.options['lstm']['projection_dim']
        if self.share_embedding_softmax:
            self.softmax_W = self.embedding_weights
        with tf.control_dependencies('softmax_W'), tf.device('/cpu:0'):
            softmax_init = tf.random_normal_initializer(0.0, 1.0/softmax_dim)
            if not self.share_embedding_softmax:
                self.softmax_W = tf.get_variable('softmax_W',
                                                 [n_tokens_vocab, softmax_dim],
                                                 dtype=tf.float32,
                                                 initializer=softmax_init)
                self.softmax_b = tf.get_variable('softmax_b',
                                                 [n_tokens_vocab],
                                                 dtype=tf.float32,
                                                 initilizer=tf.constant_initializer(0.0)
                                                 )
        self.individual_losses = []
        if self.bidirectional:
            self.next_token_ids = [self.next_token_ids, self.next_token_ids_reverse]
        else:
            self.next_token_ids = [self.next_token_ids]
        for next_id_placeholder, lstm_output_flat in zip(self.next_token_ids, lstm_outputs):
            next_token_ids_flat = tf.reshape(next_id_placeholder, [-1,1])
            with tf.control_dependencies([lstm_output_flat]):
                if self.is_training and self.sample_softmax:
                    losses = tf.nn.sampled_softmax_loss(
                        self.softmax_W, self.softmax_b,
                        next_token_ids_flat, lstm_output_flat,
                        self.options['n_negative_samples_batch'],
                        self.options['n_tokens_vocab'],
                        num_true=1
                    )
                else:
                    output_scores = tf.matmul(
                        lstm_output_flat, tf.transpose(self.softmax_W)
                    ) + self.softmax_b
                    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=output_scores,
                        labels=tf.squeeze(next_token_ids_flat, squeeze_dim=[1])
                    )
            self.individual_losses.append(tf.reduce_mean(losses))
        if self.bidirectional:
            self.total_loss = 0.5 * (self.individual_losses[0] + self.individual_losses[1])
        else:
            self.total_loss = self.individual_losses[0]



