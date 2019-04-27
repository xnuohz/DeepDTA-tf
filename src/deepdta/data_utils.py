#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import datetime
import math
import numpy as np
from sklearn.model_selection import KFold


def label_smiles(line, max_smi_len, smi_ch_ind):
    X = np.zeros(max_smi_len)
    for i, ch in enumerate(line[:max_smi_len]):
        X[i] = smi_ch_ind[ch]
    return X


def label_sequence(line, max_seq_len, smi_ch_ind):
    X = np.zeros(max_seq_len)
    for i, ch in enumerate(line[:max_seq_len]):
        X[i] = smi_ch_ind[ch]
    return X


def get_data(ligands, proteins, inter, max_smi_len, max_seq_len, char_smi_set, char_seq_set):
    inter = -np.log10(inter / math.pow(10, 9))

    smi_feature = []
    seq_feature = []

    for idx, row in ligands.iterrows():
        smi_feature.append(label_smiles(
            row[1], max_smi_len, char_smi_set))

    for idx, row in proteins.iterrows():
        seq_feature.append(label_sequence(
            row[1], max_seq_len, char_seq_set))

    return np.asarray(smi_feature), np.asarray(seq_feature), np.asarray(inter)


def get_feature(inter, smi_feature, seq_feature, train_idx, valid_idx):
    trainX, trainy, validX, validy = [], [], [], []

    for row, col in train_idx:
        trainX.append([smi_feature[row], seq_feature[col]])
        trainy.append(inter[row][col])

    for row, col in valid_idx:
        validX.append([smi_feature[row], seq_feature[col]])
        validy.append(inter[row][col])

    return np.asarray(trainX), np.reshape(trainy, [-1, 1]), np.asarray(validX), np.reshape(validy, [-1, 1])


def new_pair_fold(inter, cv_num):
    inter_row, inter_col = np.where(np.isnan(inter) == False)
    coord = np.asarray(list(zip(inter_row, inter_col)))
    kfold = KFold(n_splits=cv_num, shuffle=True, random_state=1)
    cv_train, cv_valid = [], []
    for train, valid in kfold.split(coord):
        cv_train.append(coord[train])
        cv_valid.append(coord[valid])
    return cv_train, cv_valid


def new_ligand_fold():
    pass


def new_protein_fold():
    pass


def get_now():
    return datetime.datetime.now().strftime('%y-%m-%d %H:%M:%S')
