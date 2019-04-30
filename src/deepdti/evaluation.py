#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def get_auc(target, predict):
    return roc_auc_score(target, predict)


def get_aupr(target, prediction):
    p, r, _ = precision_recall_curve(target, prediction)
    return auc(r, p)
