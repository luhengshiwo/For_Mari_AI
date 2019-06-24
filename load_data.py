#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = 'luheng'


"""
输入是一个分词后的数据集，形如[['你','好','吗'],['我','很','好','啊']]
输出是一个dataset格式的dict
"""
import time
import numpy as np
import os
import pandas as pd
from parameters import configs
import tensorflow as tf
import tensorflow.contrib as tc
import os

def build_dataset(path,shuffle_and_repeat=True):
    dataset = tf.data.TextLineDataset(path)
    if shuffle_and_repeat == True:
        dataset = dataset.shuffle(1000)
    dataset = dataset.map(
        lambda line: (tf.string_split([line]).values[1:], tf.string_split([line]).values[0]),
        num_parallel_calls=configs.num_parallel_calls
    )
    dataset = dataset.filter(lambda text, label: tf.size(text) < 1000)
    dataset = dataset.map(lambda text, label: {'text': text, 'text_len': tf.size(text), 'label': label})
    padded_shapes = {
        'text': tf.TensorShape([None]),
        'text_len': tf.TensorShape([]),
        'label': tf.TensorShape([])
    }
    padding_values = {
        'text': tf.constant(configs.pad, dtype=tf.string),
        'text_len': tf.constant(0, dtype=tf.int32),
        'label': tf.constant('0', dtype=tf.string)
    }
    dataset = dataset.apply(
        tf.data.experimental.bucket_by_sequence_length(element_length_func=lambda d: tf.size(d['text']),
                                          bucket_boundaries=[10, 30, 60, 100, 200],
                                          bucket_batch_sizes=[configs.batch_size] * 5 + [configs.batch_size // 5],
                                          padded_shapes=padded_shapes,
                                          padding_values=padding_values)
    )
    if shuffle_and_repeat == True:
        dataset = dataset.repeat(configs.n_epochs)
    return dataset

if __name__ == "__main__":
    dataset = build_dataset('data/dev4split.csv')
    print(dataset.output_shapes)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        sess.run(iterator.initializer)
        for i in range(2):
            a = sess.run(next_element)
            print(a)







