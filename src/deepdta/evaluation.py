#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def get_auc(target, predict):
    return roc_auc_score(target, predict)


def get_aupr(target, prediction):
    p, r, _ = precision_recall_curve(target, prediction)
    return auc(r, p)


def get_ci(Y, P):
    summ = 0
    pair = 0
    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if(Y[i] > Y[j]):
                    pair += 1
                    summ += 1 * (P[i] > P[j]) + 0.5 * (P[i] == P[j])

    if pair is not 0:
        return summ/pair
    else:
        return 0
