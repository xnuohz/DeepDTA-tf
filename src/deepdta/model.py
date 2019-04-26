#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from sklearn.model_selection import train_test_split
from evaluation import get_aupr, get_ci
from data_utils import get_now


class CNN(object):
    def __init__(self, filter_num, smi_window_len, seq_window_len,
                 max_smi_len, max_seq_len, char_smi_set_size, char_seq_set_size, embed_dim):
        self.smi = tf.placeholder(shape=[None, max_smi_len], dtype=tf.int32)
        self.seq = tf.placeholder(shape=[None, max_seq_len], dtype=tf.int32)
        self.labels = tf.placeholder(shape=[None, 1], dtype=tf.int32)

        self.smi_embed = tf.Variable(tf.random_normal(
            [char_smi_set_size + 1, embed_dim]))
        self.seq_embed = tf.Variable(tf.random_normal(
            [char_seq_set_size + 1, embed_dim]))

        enc_smi = tf.nn.embedding_lookup(self.smi_embed, self.smi)
        enc_smi = layers.conv1d(enc_smi, filter_num,
                                smi_window_len, padding='VALID')
        enc_smi = layers.conv1d(enc_smi, filter_num * 2,
                                smi_window_len, padding='VALID')
        enc_smi = layers.conv1d(enc_smi, filter_num * 3,
                                smi_window_len, padding='VALID')
        enc_smi = tf.keras.layers.GlobalAveragePooling1D()(enc_smi)

        enc_seq = tf.nn.embedding_lookup(self.seq_embed, self.seq)
        enc_seq = layers.conv1d(enc_seq, filter_num,
                                seq_window_len, padding='VALID')
        enc_seq = layers.conv1d(enc_seq, filter_num * 2,
                                seq_window_len, padding='VALID')
        enc_seq = layers.conv1d(enc_seq, filter_num * 3,
                                seq_window_len, padding='VALID')
        enc_seq = tf.keras.layers.GlobalAveragePooling1D()(enc_seq)

        flatten = tf.concat([enc_smi, enc_seq], -1)
        fc1 = layers.fully_connected(flatten, 1024)
        drop1 = layers.dropout(fc1, 0.1)
        fc2 = layers.fully_connected(drop1, 1024)
        drop2 = layers.dropout(fc2, 0.1)
        fc3 = layers.fully_connected(drop2, 512)

        predictions = layers.fully_connected(fc3, 1)
        cost = tf.losses.mean_squared_error(self.labels, predictions)
        print(cost.shape)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
        self.saver = tf.train.Saver()

    def train(self, sess, train_x, train_y, valid_x=None, valid_y=None, nb_epoch=None, batch_size=None,
              verbose=True, model_path=None, data_idx=None):
        print(get_now(), 'start training')
        if valid_x is None or valid_y is None:
            train_idx, valid_idx = train_test_split(
                range(len(trainx)) if data_idx is None else data_idx, test_size=0.1)
            valid_x, valid_y = train_x[valid_idx], train_y[valid_idx]
        else:
            train_idx = np.arange(
                len(train_idx)) if data_idx is None else data_idx
        sess.run(tf.global_variables())
        best_aupr = 0
        for idx in range(nb_epoch):
            np.random.shuffle(train_idx)
            train_loss, train_res = 0, np.empty(len(train_idx))
            for i in range(0, len(train_idx), batch_size):
                batch_idx = train_idx[i: i + batch_size]
                x, y = train_x[batch_idx], train_y[batch_idx]
                feed_dict = {
                    self.smi: x[:, 0],
                    self.seq: x[:, 1],
                    self.labels: y
                }
                _, loss, train_res[i: i + batch_size] = sess.run([self.optimizer, self.cost, self.predictions],
                                                                 feed_dict=feed_dict)
                train_loss += loss * len(y)
            train_loss /= len(train_idx)
            train_ci, train_aupr = get_ci(train_y[train_idx], train_res), get_aupr(
                train_y[train_idx], train_res)
            # print(train_res, trainy)

            valid_loss, valid_res = 0, np.empty(shape=valid_y.shape)
            for i in range(0, len(valid_x), batch_size):
                x, y = valid_x[i: i + batch_size], valid_y[i: i + batch_size]
                feed_dict = {
                    self.smi: x[:, 0],
                    self.seq: x[:, 1],
                    self.labels: y
                }
                loss, valid_res[i: i + batch_size] = sess.run(
                    [self.cost, self.predictions], feed_dict=feed_dict)
                valid_loss += loss * len(y)
            valid_loss /= len(valid_y)
            valid_ci, valid_aupr = get_ci(
                valid_y, valid_res), get_aupr(valid_y, valid_res)
            if verbose:
                print(get_now(), idx, "loss:", train_loss, valid_loss,
                      'CI:', train_ci, valid_ci, 'AUPR:', train_aupr, valid_aupr)
            if valid_aupr > best_aupr:
                best_aupr = valid_aupr
                self.saver.save(
                    sess, model_path if model_path is not None else 'tmp/cnn.model')

    def predict(self, sess, X, batch_size=128, model_path=None):
        print(get_now(), 'Start Predicting')
        self.saver.restore(
            sess, model_path if model_path is not None else 'tmp/cnn.model')
        res = np.empty(shape=X.shape[0])
        for i in range(0, len(X), batch_size):
            x = X[i: i + batch_size]
            feed_dict = {
                self.smi: x[:, 0],
                self.seq: x[:, 1],
                self.labels: y
            }
            res[i: i +
                batch_size] = sess.run(self.predictions, feed_dict=feed_dict)
        return res


if __name__ == "__main__":
    model = CNN(32, 4, 8, 100, 1000, 65, 26, 128)
