"""all classification model kernel"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = 'luheng'

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
# import tensorflow.estimator as te
from parameters import configs


class kernel():
    def __init__(self, embeddings, text_len, mode):
        self.embeddings = embeddings
        self.text_len = text_len
        self.mode = mode

    def lstm(self):
        pass

    def gru(self):
        train_keep_prob = 1.0
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            train_keep_prob = configs.train_keep_prob
        he_init = tc.layers.variance_scaling_initializer()
        with tf.variable_scope("cls"):
            cell = tf.nn.rnn_cell.GRUCell(
                configs.num_units, kernel_initializer=he_init)
            # cell = tf.keras.layers.GRUCell(
            #     configs.num_units, kernel_initializer=he_init)
            cell_drop = tf.nn.rnn_cell.DropoutWrapper(
                cell, input_keep_prob=train_keep_prob)
            outputs, _ = tf.nn.dynamic_rnn(
                cell_drop, self.embeddings, sequence_length=self.text_len, time_major=True, dtype=tf.float32)
        # out_state = tf.reduce_max(outputs, axis=0)
        out_state = outputs[-1,:]
        return out_state

    def bi_lstm(self):
        pass

    def bi_gru(self):
        train_keep_prob = 1.0
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            train_keep_prob = configs.train_keep_prob
        he_init = tc.layers.variance_scaling_initializer()
        with tf.variable_scope("cls"):
            forward_cell = tc.rnn.GRUCell(
                configs.num_units, kernel_initializer=he_init)
            forward_cell_drop = tc.rnn.DropoutWrapper(
                forward_cell, input_keep_prob=train_keep_prob)
            backward_cell = tc.rnn.GRUCell(
                configs.num_units, kernel_initializer=he_init)
            backward_cell_drop = tc.rnn.DropoutWrapper(
                backward_cell, input_keep_prob=train_keep_prob)
            bi_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                forward_cell_drop, backward_cell_drop, self.embeddings, sequence_length=self.text_len,
                time_major=True, dtype=tf.float32)
            outputs = tf.add_n(bi_outputs)
            out_state = tf.reduce_max(outputs, axis=0)
        return out_state

    def multi_lstm(self):
        pass

    def multi_gru(self):
        pass
