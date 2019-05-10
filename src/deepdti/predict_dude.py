#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
import configparser
import json
import os
import sys
from model import CNN, ECFPCNN
from data_utils import get_data, label_ecfp, label_sequence, label_smiles
from evaluation import get_auc, get_aupr

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


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

    ''' SMILES + seq '''
    model = CNN(filter_num=conf.getint('model', 'filter_num'),
                smi_window_len=conf.getint('model', 'smi_window_len'),
                seq_window_len=conf.getint('model', 'seq_window_len'),
                max_smi_len=max_smi_len,
                max_seq_len=max_seq_len,
                char_smi_set_size=len(char_smi_set),
                char_seq_set_size=len(char_seq_set),
                embed_dim=conf.getint('model', 'embed_dim'))
    ''' ECFP + seq '''
    # model = ECFPCNN(filter_num=conf.getint('model', 'filter_num'),
    #                 seq_window_len=conf.getint('model', 'seq_window_len'),
    #                 char_seq_set_size=len(char_seq_set),
    #                 embed_dim=conf.getint('model', 'embed_dim'),
    #                 max_smi_len=max_smi_len,
    #                 max_seq_len=max_seq_len)

    model_path = os.path.join(
        conf.get('model', 'path', fallback='tmp'), 'all.model')

    data_name = data_predicted[0]
    path = data_path + data_name + '/'

    output_path = os.path.join(path, 'output')
    if os.path.exists(output_path) is False:
        os.mkdir(output_path)

    proteins = pd.read_csv(path + 'proteins.csv', header=None)
    # ['aa2ar']
    failed_p = []
    outputs = []
    for idx, (p_name, p_seq) in proteins.iterrows():
        try:
            active_ligands_path = os.path.join(path, 'obabel_dude_smiles', 'active', '%s_smiles.csv' % p_name)
            decoy_ligands_path = os.path.join(path, 'obabel_dude_smiles', 'decoy', '%s_smiles.csv' % p_name)
            active_ligands = pd.read_csv(active_ligands_path, header=None)
            decoy_ligands = pd.read_csv(decoy_ligands_path, header=None)
            ligands = np.concatenate([active_ligands, decoy_ligands], 0)
            labels = np.concatenate([np.ones([len(active_ligands), 1]), np.zeros([len(decoy_ligands), 1])], 0)

            inputs, names = [], []
            seq_feature = label_sequence(p_seq, max_seq_len, char_seq_set)

            for l_name, l_smiles in ligands:
                names.append(l_name)
                ''' CNN '''
                smi_feature = label_smiles(l_smiles, max_smi_len, char_smi_set)
                ''' ECFPCNN '''
                # smi_feature = label_ecfp(l_smiles, max_smi_len)
                inputs.append([smi_feature, seq_feature])

            res = model.predict(sess, np.asarray(inputs), batch_size=conf.getint(
                'model', 'batch_size'), model_path=model_path)

            auc, aupr = get_auc(labels, res), get_aupr(labels, res)
            outputs.append([p_name, auc, aupr])
            final_data = pd.DataFrame(np.asarray(list(zip(names, res))))
            final_data.to_csv(os.path.join(output_path, '%s.csv' % p_name), index=None, header=None)
        except Exception:
            failed_p.append(p_name)
            continue
    pd.DataFrame(outputs).to_csv(path + 'outputs.csv', index=None, header=['target', 'auc', 'aupr'])
    print('failed', failed_p)


if __name__ == "__main__":
    main(sys.argv[1:])
