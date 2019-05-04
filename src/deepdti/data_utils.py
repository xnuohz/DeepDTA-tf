#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import datetime
import math
import numpy as np
from sklearn.model_selection import KFold
from rdkit import Chem
from rdkit.Chem import AllChem


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


def label_ecfp(line, max_smi_len):
    mol = Chem.MolFromSmiles(line)
    # radius 6
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 6, nBits=max_smi_len)
    X = list(map(int, ecfp.ToBitString()))
    return np.asarray(X)


def get_data(ligands, proteins, max_smi_len, max_seq_len, char_smi_set, char_seq_set):
    smi_feature = []
    seq_feature = []

    for idx, row in ligands.iterrows():
        smi_feature.append(label_smiles(
            row[1], max_smi_len, char_smi_set))

    for idx, row in proteins.iterrows():
        seq_feature.append(label_sequence(
            row[1], max_seq_len, char_seq_set))

    return np.asarray(smi_feature), np.asarray(seq_feature)


def get_feature(ligands, proteins, inter, coords, max_smi_len, char_smi_set, max_seq_len, char_seq_set):
    X, y = [], []
    for row, col in coords:
        smi = ligands.iloc[row, 1]
        seq = proteins.iloc[col, 1]
        try:
            smi_vector = label_smiles(smi, max_smi_len, char_smi_set)
            seq_vector = label_sequence(seq, max_seq_len, char_seq_set)
            X.append([smi_vector, seq_vector])
            y.append(inter[row][col])
        except Exception:
            continue
    return np.asarray(X), np.asarray(y).reshape([-1, 1])


def get_coord(inter):
    ''' default pos : neg = 1 : 15 '''
    pos_row, pos_col = np.where(inter == 1)
    pos_coord = np.asarray(list(zip(pos_row, pos_col)))
    neg_row, neg_col = np.where(inter == 0)
    neg_coord = np.asarray(list(zip(neg_row, neg_col)))
    select = np.random.choice(
        range(len(neg_coord)), size=15 * len(pos_coord))
    neg_coord = np.asarray(neg_coord)[select]
    return pos_coord, neg_coord


def new_pair_fold(inter, cv_num):
    pos_coord, neg_coord = get_coord(inter)
    kfold = KFold(n_splits=cv_num, shuffle=True, random_state=1)
    coord = np.concatenate([pos_coord, neg_coord], 0)
    cv_train, cv_valid = [], []
    for train, valid in kfold.split(coord):
        cv_train.append(coord[train])
        cv_valid.append(coord[valid])
    return np.asarray(cv_train), np.asarray(cv_valid)


def new_ligand_fold(inter, cv_num):
    pos_coord, neg_coord = get_coord(inter)
    kfold = KFold(n_splits=cv_num, shuffle=True, random_state=1)
    coord = np.concatenate([pos_coord, neg_coord], 0)
    cv_train, cv_valid = [], []

    d_num = np.shape(inter)[0]
    for train, valid in kfold.split(range(d_num)):
        cv_train.append(np.asarray([t for t in coord if t[0] in train]))
        cv_valid.append(np.asarray([t for t in coord if t[0] in valid]))
    return np.asarray(cv_train), np.asarray(cv_valid)


def new_protein_fold(inter, cv_num):
    pos_coord, neg_coord = get_coord(inter)
    kfold = KFold(n_splits=cv_num, shuffle=True, random_state=1)
    coord = np.concatenate([pos_coord, neg_coord], 0)
    cv_train, cv_valid = [], []
    t_num = np.shape(inter)[1]

    for train, valid in kfold.split(range(t_num)):
        cv_train.append(np.asarray([t for t in coord if t[1] in train]))
        cv_valid.append(np.asarray([t for t in coord if t[1] in valid]))
    return np.asarray(cv_train), np.asarray(cv_valid)


def get_now():
    return datetime.datetime.now().strftime('%y-%m-%d %H:%M:%S')
