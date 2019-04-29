#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
import configparser
import json
import os
import sys
from model import CNN
from data_utils import get_data


def main(argv):
    conf = configparser.ConfigParser()
    print(conf.read(argv))
    max_smi_len = conf.getint('model', 'max_smi_len')
    max_seq_len = conf.getint('model', 'max_seq_len')

    char_smi_set = json.load(open(conf.get('model', 'char_smi')))
    char_seq_set = json.load(open(conf.get('model', 'char_seq')))

    data_path = conf.get('data', 'path')
    data_predicted = conf.get('data', 'prediction').split(',')

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

    model_path = model_path = os.path.join(
        conf.get('model', 'path', fallback='tmp'), 'all.model')

    for data_name in data_predicted:
        path = data_path + data_name + '/'

        ligands = pd.read_csv(path + 'ligands.csv', header=None)
        proteins = pd.read_csv(path + 'proteins.csv', header=None)

        smi_feature, seq_feature = get_data(
            ligands, proteins, max_smi_len, max_seq_len, char_smi_set, char_seq_set)

        inputs = []
        for smif in smi_feature:
            inputs.append([smif, seq_feature[0]])
        res = model.predict(sess, np.asarray(inputs), batch_size=conf.getint(
            'model', 'batch_size'), model_path=model_path)
        names = [x.split('.')[0] for x in list(ligands.iloc[:, 0])]
        final_data = pd.DataFrame(np.asarray(list(zip(names, res))))
        final_data.to_csv(path + 'res.csv', index=None, header=None)


if __name__ == "__main__":
    main(sys.argv[1:])
