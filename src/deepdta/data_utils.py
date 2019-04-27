#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import datetime
import math
import numpy as np


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

    for d in ligands.keys():
        smi_feature.append(label_smiles(
            ligands[d], max_smi_len, char_smi_set))

    for t in proteins.keys():
        seq_feature.append(label_sequence(
            proteins[t], max_seq_len, char_seq_set))

    return np.asarray(smi_feature), np.asarray(seq_feature), np.asarray(inter)


def get_now():
    return datetime.datatime.now().strftime('%y-%m-%d %H:%M:%S')
