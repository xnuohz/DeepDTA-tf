#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import json
import os
import numpy as np
import pandas as pd
import configparser
import tensorflow as tf

from data_utils import get_now, get_data, get_feature, new_pair_fold, new_ligand_fold, new_protein_fold
from evaluation import get_aupr, get_ci
from model import CNN


def main(argv):
    conf = configparser.ConfigParser()
    print(conf.read(argv))
    max_smi_len = conf.getint('model', 'max_smi_len')
    max_seq_len = conf.getint('model', 'max_seq_len')

    data_path = conf.get('model', 'data_path')

    ligands = pd.read_csv(data_path + 'ligands.csv', header=None)
    proteins = pd.read_csv(data_path + 'proteins.csv', header=None)
    inter = pd.read_csv(data_path + 'inter.csv', header=None)
    print(ligands.shape, proteins.shape, inter.shape)

    char_smi_set = json.load(open(conf.get('model', 'char_smi')))
    char_seq_set = json.load(open(conf.get('model', 'char_seq')))

    smi_feature, seq_feature, inter = get_data(
        ligands, proteins, inter, max_smi_len, max_seq_len, char_smi_set, char_seq_set)
    print(smi_feature.shape, seq_feature.shape, inter.shape)

    cv_num = conf.getint('cv', 'cv_num', fallback=5)
    cv_train, cv_valid = new_pair_fold(inter, cv_num)

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
        model_path = os.path.join(
            conf.get('model', 'path', fallback='tmp'), 'cv-' + str(cv_id) + '.model')
        trainX, trainy, validX, validy = get_feature(
            inter, smi_feature, seq_feature, cv_train[cv_id], cv_valid[cv_id])
        model.train(sess, trainX, trainy, validX, validy,
                    nb_epoch=conf.getint('model', 'num_epoch'),
                    batch_size=conf.getint('model', 'batch_size'),
                    model_path=model_path)


if __name__ == "__main__":
    main(sys.argv[1:])
