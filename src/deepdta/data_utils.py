#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import datetime


def one_hot_smiles(line, max_smi_len, smi_ch_ind):
    x = np.zeros((max_smi_len, len(smi_ch_ind)))
    for i, ch in enumerate(line[:max_smi_len]):
        x[i, (smi_ch_ind[ch]-1)] = 1
    return x


def one_hot_sequence(line, max_seq_len, smi_ch_ind):
    X = np.zeros((max_seq_len, len(smi_ch_ind)))
    for i, ch in enumerate(line[:max_seq_len]):
        X[i, (smi_ch_ind[ch])-1] = 1
    return X


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


def get_now():
    return datetime.datatime.now().strftime('%y-%m-%d %H:%M:%S')
