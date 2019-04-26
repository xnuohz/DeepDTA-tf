#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
import json
import numpy as np
import configparser
import tensorflow as tf

from data_utils import get_now
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
    char_smi = conf.get('model', 'char_smi')
    char_seq = conf.get('model', 'char_seq')
    print(char_seq, embed_dim)


if __name__ == "__main__":
    main(sys.argv[1:])
