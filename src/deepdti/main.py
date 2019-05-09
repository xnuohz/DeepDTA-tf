#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import json
import os
import numpy as np
import pandas as pd
import configparser
import tensorflow as tf

from data_utils import get_now, label_smiles, label_sequence, get_coord, label_ecfp, get_feature
from model import CNN, ECFPCNN

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


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
    inter = np.asarray(inter)
    print(ligands.shape, proteins.shape, inter.shape)

    char_smi_set = json.load(open(conf.get('model', 'char_smi')))
    char_seq_set = json.load(open(conf.get('model', 'char_seq')))

    sess = tf.InteractiveSession(
        config=tf.ConfigProto(allow_soft_placement=True))
    ''' SMILES + seq '''
    # model = CNN(filter_num=conf.getint('model', 'filter_num'),
    #             smi_window_len=conf.getint('model', 'smi_window_len'),
    #             seq_window_len=conf.getint('model', 'seq_window_len'),
    #             max_smi_len=max_smi_len,
    #             max_seq_len=max_seq_len,
    #             char_smi_set_size=len(char_smi_set),
    #             char_seq_set_size=len(char_seq_set),
    #             embed_dim=conf.getint('model', 'embed_dim'))
    ''' ECFP + seq '''
    model = ECFPCNN(filter_num=conf.getint('model', 'filter_num'),
                    seq_window_len=conf.getint('model', 'seq_window_len'),
                    char_seq_set_size=len(char_seq_set),
                    embed_dim=conf.getint('model', 'embed_dim'),
                    max_smi_len=max_smi_len,
                    max_seq_len=max_seq_len)

    pos_coord, neg_coord = get_coord(inter)
    coords = np.concatenate([pos_coord, neg_coord], 0)
    trainX, trainy = get_feature(ligands, proteins, inter, coords, max_smi_len, char_smi_set, max_seq_len, char_seq_set)

    print(trainX.shape, trainy.shape)

    model_path = os.path.join(
        conf.get('model', 'path', fallback='tmp'), 'all.model')
    model.train(sess, trainX, trainy,
                nb_epoch=conf.getint('model', 'num_epoch'),
                batch_size=conf.getint('model', 'batch_size'),
                model_path=model_path)


if __name__ == "__main__":
    main(sys.argv[1:])
