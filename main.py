"""An Example of a custom Estimator for the dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tf_metrics import precision, recall, f1
import tensorflow.contrib as tc
from parameters import configs
from methods import kernel
from load_data import build_dataset
import functools
from pathlib import Path
import logging
import sys

vocab_path = 'data/vocab.txt'
train_path = 'data/train.csv'
dev_path = 'data/train.csv'
vector = 'data/word2vec.npy'

# Logging
Path('results').mkdir(exist_ok=True)
#DEBUG
tf.logging.set_verbosity(logging.INFO)
handlers = [
    logging.FileHandler('results/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers


def model_fn(features, mode):
    # For serving, features are a bit different
    if isinstance(features, dict):
        features = features['text'], features['text_len'], features['label']
    #创建look_up table,可以将word转化为id,需要手动设置default_value=3374
    word_table = tc.lookup.index_table_from_file(vocab_path,default_value=3374)
    label_table = tc.lookup.index_table_from_tensor(['0','1','2'])
    # Read vocabs and inputs
    words_, nwords, label_ = features
    #利用table将word转化为id
    words = word_table.lookup(words_)
    label = label_table.lookup(label_)

    training = (mode == tf.estimator.ModeKeys.TRAIN)
    vec = np.load(vector)
    word_embeddings = tf.Variable(vec, trainable=True, name="embeddings")
    embeddings = tf.nn.embedding_lookup(word_embeddings, words)
    embeddings = tf.layers.dropout(embeddings, rate=configs.train_keep_prob, training=training)

    # bi_gru
    embeddings_trans = tf.transpose(embeddings, perm=[1, 0, 2])
    predict_method = kernel(embeddings_trans, nwords, mode)
    output = predict_method.bi_gru()

    # softmax
    logits = tf.layers.dense(output, configs.n_outputs)
    confidence = tf.reduce_max(tf.nn.softmax(logits, axis=-1), axis=-1)
    pred = tf.argmax(logits, axis=-1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions
        predictions = {
            'pred': pred,
            'confidence': confidence
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Loss
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label, logits=logits)
        base_loss = tf.reduce_mean(xentropy)
        # regularization
        reg_losses = tc.layers.apply_regularization(
            tc.layers.l2_regularizer(configs.l2_rate), tf.trainable_variables())
        loss = base_loss + reg_losses

        metrics = {
            'acc': tf.metrics.accuracy(label, pred),
            'precision': precision(label, pred, configs.n_outputs),
            'recall': recall(label, pred, configs.n_outputs),
            'f1': f1(label, pred, configs.n_outputs),
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            # 梯度剪裁
            optimizer = tf.train.AdamOptimizer(
                learning_rate=configs.learning_rate)
            grads_and_vars = optimizer.compute_gradients(loss)
            capped_gvs = [(tf.clip_by_value(grad, -configs.threshold, configs.threshold), var)
                          for grad, var in grads_and_vars]
            train_op = optimizer.apply_gradients(
                capped_gvs, global_step=tf.train.get_or_create_global_step())
            # train_op = tf.train.AdamOptimizer().minimize(
            #     loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)


def serving_input_receiver_fn():
    text = tf.placeholder(dtype=tf.string, shape=[None, None], name='text')
    text_len = tf.placeholder(dtype=tf.int32, shape=[None], name='text_len')
    label = tf.placeholder(dtype=tf.string, shape=[None], name='label')
    receiver_tensors = {'text': text, 'text_len': text_len, 'label': label}
    features = {'text': text, 'text_len': text_len, 'label': label}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def train():
    cfg = tf.estimator.RunConfig(save_checkpoints_steps=140)
    estimator = tf.estimator.Estimator(model_fn, 'results/model', cfg)
    train_inpf = functools.partial(build_dataset,train_path)
    # train_for_evaluate = functools.partial(build_dataset,train_path,False)
    dev_inpf = functools.partial(build_dataset,dev_path,False)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    hook = tf.contrib.estimator.stop_if_no_increase_hook(
        estimator, 'f1', 140, min_steps=1400, run_every_secs=None, run_every_steps=14)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, max_steps=2000, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=dev_inpf, throttle_secs=120)
    # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    estimator.train(input_fn=train_inpf, max_steps=15100, hooks=[hook])
    # estimator.evaluate(input_fn=train_inpf)
    estimator.evaluate(input_fn=dev_inpf)
    estimator.export_savedmodel(
        'saved_model', serving_input_receiver_fn=serving_input_receiver_fn)

def write_predictions(name):
    Path('results/score').mkdir(parents=True, exist_ok=True)
    with Path('results/score/{}.preds.txt'.format(name)).open('wb') as f:
        test_inpf = functools.partial(build_dataset,dev_path,False)
        # golds_gen = generator_fn(fwords(name), ftags(name))
        estimator = tf.estimator.Estimator(model_fn, 'results/model')
        preds_gen = estimator.predict(test_inpf)
        for preds in preds_gen:
            print(preds)

if __name__ == '__main__':
    train()
    # write_predictions('test')
