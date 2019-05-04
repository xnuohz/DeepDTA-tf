#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from sklearn.model_selection import train_test_split
from data_utils import get_now
from evaluation import get_auc, get_aupr


class BaseModel(object):
    def __init__(self, max_seq_len):
        self.seq = tf.placeholder(shape=[None, max_seq_len], dtype=tf.int32)
        self.labels = tf.placeholder(shape=[None, 1], dtype=tf.int32)
        self.init = tf.global_variables_initializer

    def build_smiles(self, char_smi_set_size, embed_dim, filter_num, smi_window_len):
        self.smi_embed = tf.Variable(tf.random_normal(
            [char_smi_set_size + 1, embed_dim]))
        enc_smi = tf.nn.embedding_lookup(self.smi_embed, self.smi)
        enc_smi = layers.conv1d(enc_smi, filter_num,
                                smi_window_len, padding='VALID')
        enc_smi = layers.conv1d(enc_smi, filter_num * 2,
                                smi_window_len, padding='VALID')
        enc_smi = layers.conv1d(enc_smi, filter_num * 3,
                                smi_window_len, padding='VALID')
        enc_smi = tf.keras.layers.GlobalAveragePooling1D()(enc_smi)
        return enc_smi

    def build_ecfp(self):
        fc1 = layers.fully_connected(self.smi, 1024)
        drop1 = layers.dropout(fc1, 0.1)
        fc2 = layers.fully_connected(drop1, 1024)
        drop2 = layers.dropout(fc2, 0.1)
        fc3 = layers.fully_connected(drop2, 512)
        return fc3

    def build_sequence(self, char_seq_set_size, embed_dim, filter_num, seq_window_len):
        self.seq_embed = tf.Variable(tf.random_normal(
            [char_seq_set_size + 1, embed_dim]))
        enc_seq = tf.nn.embedding_lookup(self.seq_embed, self.seq)
        enc_seq = layers.conv1d(enc_seq, filter_num,
                                seq_window_len, padding='VALID')
        enc_seq = layers.conv1d(enc_seq, filter_num * 2,
                                seq_window_len, padding='VALID')
        enc_seq = layers.conv1d(enc_seq, filter_num * 3,
                                seq_window_len, padding='VALID')
        enc_seq = tf.keras.layers.GlobalAveragePooling1D()(enc_seq)
        return enc_seq

    def train(self, sess, train_x, train_y, valid_x=None, valid_y=None, nb_epoch=None, batch_size=None,
              verbose=True, model_path=None, data_idx=None):
        print(get_now(), 'start training')
        if valid_x is None or valid_y is None:
            train_idx, valid_idx = train_test_split(
                range(len(train_x)) if data_idx is None else data_idx, test_size=0.1)
            valid_x, valid_y = train_x[valid_idx], train_y[valid_idx]
        else:
            train_idx = np.arange(
                len(train_x)) if data_idx is None else data_idx
        sess.run(self.init())

        best_aupr = 0
        for idx in range(nb_epoch):
            np.random.shuffle(train_idx)
            train_loss, train_res = 0, np.empty(len(train_idx))
            for i in range(0, len(train_idx), batch_size):
                batch_idx = train_idx[i: i + batch_size]
                x, y = train_x[batch_idx], train_y[batch_idx]
                # self.smi: [?, 100], self.seq: [?, 1000]
                feed_dict = {
                    self.smi: np.asarray([t[0] for t in x]),
                    self.seq: np.asarray([t[1] for t in x]),
                    self.labels: y
                }
                # preds: [?, 1], train_res[i: i + batch_size]: [?,]
                _, loss, preds = sess.run([self.optimizer, self.cost, self.predictions],
                                          feed_dict=feed_dict)
                train_res[i: i + batch_size] = np.squeeze(preds, 1)
                train_loss += loss * len(y)
            train_loss /= len(train_idx)
            train_auc, train_aupr = get_auc(train_y[train_idx], train_res), get_aupr(
                train_y[train_idx], train_res)

            valid_loss, valid_res = 0, np.empty(shape=valid_y.shape)
            for i in range(0, len(valid_x), batch_size):
                x, y = valid_x[i: i + batch_size], valid_y[i: i + batch_size]
                feed_dict = {
                    self.smi: np.asarray([t[0] for t in x]),
                    self.seq: np.asarray([t[1] for t in x]),
                    self.labels: y
                }

                loss, valid_res[i: i + batch_size] = sess.run(
                    [self.cost, self.predictions], feed_dict=feed_dict)

                valid_loss += loss * len(y)
            valid_loss /= len(valid_y)
            valid_auc, valid_aupr = get_auc(
                valid_y, valid_res), get_aupr(valid_y, valid_res)
            if verbose:
                print(get_now(), idx, "loss:", round(train_loss, 4), round(
                    valid_loss, 4), 'AUC:', train_auc, valid_auc, 'AUPR:', train_aupr, valid_aupr)
            if valid_aupr > best_aupr:
                best_aupr = valid_aupr
                self.saver.save(
                    sess, model_path if model_path is not None else 'tmp/cnn-classifier.model')

    def predict(self, sess, X, model_path, batch_size=128):
        assert model_path is not None
        print(get_now(), 'Start Predicting')
        # variables should also be initialized in prediction process!
        sess.run(self.init())
        self.saver.restore(sess, model_path)
        res = np.empty(shape=X.shape[0])
        for i in range(0, len(X), batch_size):
            x = X[i: i + batch_size]
            feed_dict = {
                self.smi: np.asarray([t[0] for t in x]),
                self.seq: np.asarray([t[1] for t in x])
            }
            preds = sess.run(self.predictions, feed_dict=feed_dict)
            res[i: i + batch_size] = np.squeeze(preds, 1)
        return res


class CNN(BaseModel):
    ''' SMILES + Sequence '''

    def __init__(self, filter_num, smi_window_len, seq_window_len, char_smi_set_size, char_seq_set_size, embed_dim, max_smi_len, **kwargs):
        super(CNN, self).__init__(**kwargs)
        self.smi = tf.placeholder(shape=[None, max_smi_len], dtype=tf.int32)
        enc_smi = self.build_smiles(
            char_smi_set_size, embed_dim, filter_num, smi_window_len)
        enc_seq = self.build_sequence(
            char_seq_set_size, embed_dim, filter_num, seq_window_len)

        flatten = tf.concat([enc_smi, enc_seq], -1)
        fc1 = layers.fully_connected(flatten, 1024)
        drop1 = layers.dropout(fc1, 0.1)
        fc2 = layers.fully_connected(drop1, 1024)
        drop2 = layers.dropout(fc2, 0.1)
        fc3 = layers.fully_connected(drop2, 512)

        self.predictions = layers.fully_connected(
            fc3, 1, activation_fn=tf.nn.sigmoid)
        self.cost = tf.losses.log_loss(self.labels, self.predictions)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)
        self.saver = tf.train.Saver()


class ECFPCNN(BaseModel):
    ''' ECFP + Sequence '''

    def __init__(self, filter_num, seq_window_len, char_seq_set_size, embed_dim, max_smi_len, **kwargs):
        super(ECFPCNN, self).__init__(**kwargs)
        self.smi = tf.placeholder(shape=[None, max_smi_len], dtype=tf.float32)
        # smi encode params is fixed currently.
        enc_smi = self.build_ecfp()
        enc_seq = self.build_sequence(
            char_seq_set_size, embed_dim, filter_num, seq_window_len)

        flatten = tf.concat([enc_smi, enc_seq], -1)
        fc1 = layers.fully_connected(flatten, 1024)
        drop1 = layers.dropout(fc1, 0.1)
        fc2 = layers.fully_connected(drop1, 1024)
        drop2 = layers.dropout(fc2, 0.1)
        fc3 = layers.fully_connected(drop2, 512)

        self.predictions = layers.fully_connected(
            fc3, 1, activation_fn=tf.nn.sigmoid)
        self.cost = tf.losses.log_loss(self.labels, self.predictions)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)
        self.saver = tf.train.Saver()
