#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import json
import os
import numpy as np
import pandas as pd
import configparser
import tensorflow as tf

from sklearn.model_selection import KFold
from data_utils import get_now, label_smiles, label_sequence
from model import CNN

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def main(argv):
    conf = configparser.ConfigParser()
    print(conf.read(argv))
    max_smi_len = conf.getint('model', 'max_smi_len')
    max_seq_len = conf.getint('model', 'max_seq_len')

    data_path = conf.get('model', 'data_path')

    proteins = pd.read_csv(data_path + 'proteins.csv', header=None)

    char_smi_set = json.load(open(conf.get('model', 'char_smi')))
    char_seq_set = json.load(open(conf.get('model', 'char_seq')))

    cv_num = conf.getint('cv', 'cv_num', fallback=5)
    kfold = KFold(n_splits=cv_num, shuffle=True, random_state=1)

    cv_train, cv_valid = [], []
    for train, valid in kfold.split(np.arange(len(proteins))):
        cv_train.append(train)
        cv_valid.append(valid)
    cv_train, cv_valid = np.asarray(cv_train), np.asarray(cv_valid)

    sess = tf.InteractiveSession(
        config=tf.ConfigProto(allow_soft_placement=True))
    model = CNN(filter_num=conf.getint('model', 'filter_num'),
                smi_window_len=conf.getint('model', 'smi_window_len'),
                seq_window_len=conf.getint('model', 'seq_window_len'),
                max_smi_len=max_smi_len,
                max_seq_len=max_seq_len,
                char_smi_set_size=len(char_smi_set),
                char_seq_set_size=len(char_seq_set),
                embed_dim=conf.getint('model', 'embed_dim'))

    for cv_id in range(cv_num):
        print('start cv', cv_id)
        model_path = os.path.join(
            conf.get('model', 'path', fallback='tmp'), 'cv-' + str(cv_id) + '.model')
        trainX, trainy, trainE = get_dude_feature(
            data_path, proteins, cv_train[cv_id], max_smi_len, char_smi_set, max_seq_len, char_seq_set)
        validX, validy, validE = get_dude_feature(
            data_path, proteins, cv_valid[cv_id], max_smi_len, char_smi_set, max_seq_len, char_seq_set)
        print(trainX.shape, trainy.shape, validX.shape, validy.shape)
        print('train error smiles ', trainE, 'valid error smiles ', validE)
        model.train(sess, trainX, trainy, validX, validy,
                    nb_epoch=conf.getint('model', 'num_epoch'),
                    batch_size=conf.getint('model', 'batch_size'),
                    model_path=model_path)
        break


def get_dude_feature(path, proteins, coords, max_smi_len, char_smi_set, max_seq_len, char_seq_set):
    X, y = [], []
    error_smiles = 0
    for index in coords:
        p_name, p_seq = proteins.iloc[index, :]

        active_ligands_path = os.path.join(path, 'obabel_dude_smiles', 'active', '%s_smiles.csv' % p_name)
        decoy_ligands_path = os.path.join(path, 'obabel_dude_smiles', 'decoy', '%s_smiles.csv' % p_name)
        active_ligands = pd.read_csv(active_ligands_path, header=None)
        decoy_ligands = pd.read_csv(decoy_ligands_path, header=None)

        seq_feature = label_sequence(p_seq, max_seq_len, char_seq_set)
        for _, l_smiles in np.asarray(active_ligands):
            try:
                smi_feature = label_smiles(l_smiles, max_smi_len, char_smi_set)
                X.append([smi_feature, seq_feature])
                y.append(1)
            except Exception:
                error_smiles += 1
                continue
        for _, l_smiles in np.asarray(decoy_ligands):
            try:
                smi_feature = label_smiles(l_smiles, max_smi_len, char_smi_set)
                X.append([smi_feature, seq_feature])
                y.append(0)
            except Exception:
                error_smiles += 1
                continue

    return np.asarray(X), np.asarray(y).reshape([-1, 1]), error_smiles


if __name__ == "__main__":
    main(sys.argv[1:])
