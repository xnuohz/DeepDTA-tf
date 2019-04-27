#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import numpy as np
import json
import pandas as pd
import pickle
from data_utils import label_sequence, label_smiles
from collections import OrderedDict


CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21,
               "V": 22, "Y": 23, "X": 24,
               "Z": 25}

CHARPROTLEN = 25

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64


class Dataset(object):
    def __init__(self, fpath, setting_no, seqlen, smilen, need_shuffle=False):
        self.SEQLEN = seqlen
        self.SMILEN = smilen
        self.charseqset = CHARPROTSET
        self.charseqset_size = CHARPROTLEN

        self.charsmiset = CHARISOSMISET
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

        print(len(test_fold))
        print(np.shape(train_folds))

        return test_fold, train_folds

    def parse_data(self, fpath):

        print("Read %s start" % fpath)

        ligands = json.load(open(fpath+"ligands_can.txt"),
                            object_pairs_hook=OrderedDict)
        print(ligands)
        proteins = json.load(open(fpath+"proteins.txt"),
                             object_pairs_hook=OrderedDict)

        Y = pickle.load(open(fpath + "Y", "rb"),
                        encoding='latin1')  # TODO: read from raw

        Y = -(np.log10(Y/(math.pow(10, 9))))

        XD = []
        XT = []

        for d in ligands.keys():
            XD.append(label_smiles(
                ligands[d], self.SMILEN, self.charsmiset))

        for t in proteins.keys():
            XT.append(label_sequence(
                proteins[t], self.SEQLEN, self.charseqset))

        print(np.shape(XD))

        return XD, XT, Y


if __name__ == "__main__":
    dataset = Dataset('../../data/kiba/', 1, 1000, 100)
    dataset.parse_data('../../data/kiba/')
