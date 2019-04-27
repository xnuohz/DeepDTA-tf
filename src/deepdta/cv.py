#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import json
import numpy as np
import pandas as pd
import configparser
import tensorflow as tf

from data_utils import get_now, get_data
from evaluation import get_aupr, get_ci
from model import CNN


def main(argv):
    conf = configparser.ConfigParser()
    print(conf.read(argv))
    filter_num = conf.getint('model', 'filter_num')
    seq_window_len = conf.getint('model', 'seq_window_len')
    smi_window_len = conf.getint('model', 'smi_window_len')
    max_smi_len = conf.getint('model', 'max_smi_len')
    max_seq_len = conf.getint('model', 'max_seq_len')
    embed_dim = conf.getint('model', 'embed_dim')
    batch_size = conf.getint('model', 'batch_size')
    num_epoch = conf.getint('model', 'num_epoch')

    data_path = conf.get('model', 'data_path')

    ligands = pd.read_csv(data_path + 'ligands.csv', index_col=0, header=None)
    proteins = pd.read_csv(data_path + 'proteins.csv',
                           index_col=0, header=None)
    inter = pd.read_csv(data_path + 'inter.csv', header=None)
    print(ligands.shape, proteins.shape, inter.shape)

    char_smi_set = json.load(open(conf.get('model', 'char_smi')))
    char_seq_set = json.load(open(conf.get('model', 'char_seq')))

    smi_feature, seq_feature, inter = get_data(
        ligands, proteins, inter, max_smi_len, max_seq_len, char_smi_set, char_seq_set)
    print(smi_feature.shape, seq_feature.shape, inter.shape)


if __name__ == "__main__":
    main(sys.argv[1:])
