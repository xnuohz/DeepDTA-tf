#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import numpy as np
import json
import pickle
from collections import OrderedDict


class Dataset(object):
    def __init__(self, fpath, setting_no, seqlen, smilen, need_shuffle=False):
        self.SEQLEN = seqlen
        self.SMILEN = smilen
        self.charseqset = CHARPROTSET
        self.charseqset_size = CHARPROTLEN

        self.charsmiset = CHARISOSMISET  # HERE CAN BE EDITED
        self.charsmiset_size = CHARISOSMILEN
        self.PROBLEMSET = setting_no

        # read raw file
        self._raw = self.read_sets(fpath, setting_no)

        # iteration flags
        self._num_data = len(self._raw)

    def read_sets(self, fpath, setting_no):
        print("Reading %s start" % fpath)

        test_fold = json.load(
            open(fpath + "folds/test_fold_setting" + str(setting_no)+".txt"))
        train_folds = json.load(
            open(fpath + "folds/train_fold_setting" + str(setting_no)+".txt"))

        return test_fold, train_folds

    def parse_data(self, fpath,  with_label=True):

        print("Read %s start" % fpath)

        ligands = json.load(open(fpath+"ligands_can.txt"),
                            object_pairs_hook=OrderedDict)
        proteins = json.load(open(fpath+"proteins.txt"),
                             object_pairs_hook=OrderedDict)

        Y = pickle.load(open(fpath + "Y", "rb"),
                        encoding='latin1')  # TODO: read from raw
        # if FLAGS.is_log:
        Y = -(np.log10(Y/(math.pow(10, 9))))

        XD = []
        XT = []

        if with_label:
            for d in ligands.keys():
                XD.append(label_smiles(
                    ligands[d], self.SMILEN, self.charsmiset))

            for t in proteins.keys():
                XT.append(label_sequence(
                    proteins[t], self.SEQLEN, self.charseqset))
        else:
            for d in ligands.keys():
                XD.append(one_hot_smiles(
                    ligands[d], self.SMILEN, self.charsmiset))

            for t in proteins.keys():
                XT.append(one_hot_sequence(
                    proteins[t], self.SEQLEN, self.charseqset))

        return XD, XT, Y
