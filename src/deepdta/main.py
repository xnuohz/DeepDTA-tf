#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import json
import os
import numpy as np
import pandas as pd
import configparser
import tensorflow as tf

from data_utils import get_now, get_data
from model import CNN

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


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
    pairs = pd.read_csv(data_path + 'pairs.csv', header=None)

    print(ligands.shape, proteins.shape, pairs.shape)

    char_smi_set = json.load(open(conf.get('model', 'char_smi')))
    char_seq_set = json.load(open(conf.get('model', 'char_seq')))

    smi_feature, seq_feature = get_data(
        ligands, proteins, max_smi_len, max_seq_len, char_smi_set, char_seq_set)
    print(smi_feature.shape, seq_feature.shape)

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

    trainy = np.asarray(pairs.iloc[:, 2]).reshape([-1, 1])
    trainX = []
    for idx, row in pairs.iterrows():
        ligand_index = ligands[ligands.id == row[0]].index.values[0]
        protein_index = proteins[proteins.id == row[1]].index.values[0]
        trainX.append([smi_feature[ligand_index], seq_feature[protein_index]])
    trainX = np.asarray(trainX)

    model_path = os.path.join(
        conf.get('model', 'path', fallback='tmp'), 'all.model')
    model.train(sess, trainX, trainy,
                nb_epoch=conf.getint('model', 'num_epoch'),
                batch_size=conf.getint('model', 'batch_size'),
                model_path=model_path)


if __name__ == "__main__":
    main(sys.argv[1:])
