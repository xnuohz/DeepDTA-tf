#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import json
import os
import numpy as np
import pandas as pd
import configparser
import tensorflow as tf

from data_utils import get_now, get_feature, new_pair_fold, new_ligand_fold, new_protein_fold
from model import CNN, ECFPCNN

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def create_folds(argv):
    conf = configparser.ConfigParser()
    print(conf.read(argv))

    data_path = conf.get('model', 'data_path')

    inter = pd.read_csv(data_path + 'inter.csv', header=None)
    inter = np.asarray(inter)

    cv_num = conf.getint('cv', 'cv_num', fallback=5)

    problem_type = conf.getint('cv', 'problem_type', fallback=1)
    if problem_type == 1:
        cv_train, cv_valid = new_pair_fold(inter, cv_num)
        np.save('../folds/p_train.npy', cv_train)
        np.save('../folds/p_valid.npy', cv_valid)
    elif problem_type == 2:
        cv_train, cv_valid = new_protein_fold(inter, cv_num)
        np.save('../folds/t_train.npy', cv_train)
        np.save('../folds/t_valid.npy', cv_valid)
    elif problem_type == 3:
        cv_train, cv_valid = new_ligand_fold(inter, cv_num)
        np.save('../folds/d_train.npy', cv_train)
        np.save('../folds/d_valid.npy', cv_valid)


def main(argv):
    conf = configparser.ConfigParser()
    print(conf.read(argv))
    max_smi_len = conf.getint('model', 'max_smi_len')
    max_seq_len = conf.getint('model', 'max_seq_len')

    data_path = conf.get('model', 'data_path')

    ligands = pd.read_csv(data_path + 'ligands.csv', header=None)
    proteins = pd.read_csv(data_path + 'proteins.csv', header=None)
    inter = pd.read_csv(data_path + 'inter.csv', header=None)
    inter = np.asarray(inter)
    print(ligands.shape, proteins.shape, inter.shape)

    char_smi_set = json.load(open(conf.get('model', 'char_smi')))
    char_seq_set = json.load(open(conf.get('model', 'char_seq')))

    cv_num = conf.getint('cv', 'cv_num', fallback=5)

    problem_type = conf.getint('cv', 'problem_type', fallback=1)
    if problem_type == 1:
        cv_train = np.load('../folds/p_train.npy')
        cv_valid = np.load('../folds/p_valid.npy')
    elif problem_type == 2:
        cv_train = np.load('../folds/t_train.npy')
        cv_valid = np.load('../folds/t_valid.npy')
    elif problem_type == 3:
        cv_train = np.load('../folds/d_train.npy')
        cv_valid = np.load('../folds/d_valid.npy')

    print(cv_train[0].shape, cv_valid[0].shape)

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

    for cv_id in range(cv_num):
        print('start cv', cv_id)
        model_path = os.path.join(
            conf.get('model', 'path', fallback='tmp'), 'cv-' + str(cv_id) + '.model')
        trainX, trainy = get_feature(
            ligands, proteins, inter, cv_train[cv_id], max_smi_len, char_smi_set, max_seq_len, char_seq_set)
        validX, validy = get_feature(
            ligands, proteins, inter, cv_valid[cv_id], max_smi_len, char_smi_set, max_seq_len, char_seq_set)
        print(trainX.shape, trainy.shape, validX.shape, validy.shape)
        model.train(sess, trainX, trainy, validX, validy,
                    nb_epoch=conf.getint('model', 'num_epoch'),
                    batch_size=conf.getint('model', 'batch_size'),
                    model_path=model_path)
        break


if __name__ == "__main__":
    # 生成3种预测类型的划分 python cv_compare.py ../../config/model.cfg ../../config/data_drugbank.cfg
    # create_folds(sys.argv[1:])
    main(sys.argv[1:])
