#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import json
import os
import numpy as np
import pandas as pd
import configparser
import tensorflow as tf

from data_utils import get_now, label_smiles, label_sequence, get_coord
from model import CNN

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(argv):
    conf = configparser.ConfigParser()
    print(conf.read(argv))
    max_smi_len = conf.getint('model', 'max_smi_len')
    max_seq_len = conf.getint('model', 'max_seq_len')

    data_path = conf.get('model', 'data_path')

    ligands = pd.read_csv(data_path + 'ligands.csv',
                          header=None, names=['id', 'smi'])
    proteins = pd.read_csv(data_path + 'proteins.csv',
                           header=None, names=['id', 'seq'])
    inter = pd.read_csv(data_path + 'inter.csv', header=None)

    print(ligands.shape, proteins.shape, inter.shape)

    char_smi_set = json.load(open(conf.get('model', 'char_smi')))
    char_seq_set = json.load(open(conf.get('model', 'char_seq')))

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

    trainX, trainy = [], []

    pos_coord, neg_coord = get_coord(inter)
    # pos
    for row, col in pos_coord:
        smi = ligands.iloc[row, 1]
        seq = proteins.iloc[col, 1]
        try:
            smi_vector = label_smiles(smi, max_smi_len, char_smi_set)
            seq_vector = label_sequence(seq, max_seq_len, char_seq_set)
            trainX.append([smi_vector, seq_vector])
            trainy.append(1)
        except Exception:
            continue
    # neg
    for row, col in neg_coord:
        smi = ligands.iloc[row, 1]
        seq = proteins.iloc[col, 1]
        try:
            smi_vector = label_smiles(smi, max_smi_len, char_smi_set)
            seq_vector = label_sequence(seq, max_seq_len, char_seq_set)
            trainX.append([smi_vector, seq_vector])
            trainy.append(0)
        except Exception:
            continue

    trainX = np.asarray(trainX)
    trainy = np.asarray(trainy).reshape([-1, 1])

    print(trainX.shape, trainy.shape)

    model_path = os.path.join(
        conf.get('model', 'path', fallback='tmp'), 'all.model')
    model.train(sess, trainX, trainy,
                nb_epoch=conf.getint('model', 'num_epoch'),
                batch_size=conf.getint('model', 'batch_size'),
                model_path=model_path)


if __name__ == "__main__":
    main(sys.argv[1:])
