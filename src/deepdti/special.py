#!/usr/bin/env python3
# -*- coding:utf-8 -*-
''' 删除第540列，用其他的数据训练，540列预测 '''
import sys
import json
import os
import numpy as np
import pandas as pd
import configparser
import tensorflow as tf

from data_utils import get_now, get_coord, label_smiles, label_sequence, label_ecfp, get_feature
from model import CNN, ECFPCNN
from evaluation import get_auc, get_aupr

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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

    other_inter = inter.drop(axis=1, columns=539)

    inter = np.asarray(inter)
    other_inter = np.asarray(other_inter)

    print(ligands.shape, proteins.shape, inter.shape, other_inter.shape)

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

    pred_coords = np.concatenate([np.arange(1567).reshape([-1, 1]), np.asarray([539] * 1567).reshape([-1, 1])], 1)
    predX, predy = get_feature(ligands, proteins, inter, pred_coords, max_smi_len, char_smi_set, max_seq_len, char_seq_set)

    print(predX.shape)

    model_path = os.path.join(
        conf.get('model', 'path', fallback='tmp'), 'all.model')
    model.train(sess, trainX, trainy,
                nb_epoch=conf.getint('model', 'num_epoch'),
                batch_size=conf.getint('model', 'batch_size'),
                model_path=model_path)
    res = model.predict(sess, predX, batch_size=conf.getint(
        'model', 'batch_size'), model_path=model_path)

    print('pred', list(res))
    print('truth', predy)
    print('AUC: ', get_auc(predy, res))
    print('AUPR: ', get_aupr(predy, res))


if __name__ == "__main__":
    main(sys.argv[1:])
