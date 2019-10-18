import os
import csv
from collections import defaultdict
from glob import glob
from datetime import datetime
from multiprocessing import Manager, freeze_support, Process
import numpy as np
import scipy.stats
from scipy.special import psi, polygamma
from sklearn.metrics import roc_auc_score

from sklearn.svm import OneClassSVM
from sklearn.model_selection import ParameterGrid
from sklearn.externals.joblib import Parallel, delayed
from keras.models import Model, Input, Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

from utils import load_cifar10, load_cats_vs_dogs, load_fashion_mnist, load_cifar100, load_mnist
from utils import save_roc_pr_curve_data, get_class_name_from_index, get_channels_axis#, save_transformation

from transformations import Transformer
from models.wide_residual_network import create_wide_residual_network
from models.encoders_decoders import conv_encoder, conv_decoder
from models import dsebm, dagmm, adgan
import keras.backend as K
from keras.callbacks import ModelCheckpoint

import argparse
from argparse import Namespace


import fnmatch
import os



RESULTS_DIR = './results/' # saved weights


def create_auc_table(metric='roc_auc'):
    file_path = glob(os.path.join(RESULTS_DIR, '*', '*.npz'))
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    methods = set()
    for p in file_path:
        _, f_name = os.path.split(p)
        dataset_name, method, single_class_name = f_name.split(sep='_')[:3]
        methods.add(method)
        npz = np.load(p)
        roc_auc = npz[metric]
        results[dataset_name][single_class_name][method].append(roc_auc)

    for ds_name in results:
        for sc_name in results[ds_name]:
            for method_name in results[ds_name][sc_name]:
                roc_aucs = results[ds_name][sc_name][method_name]
                results[ds_name][sc_name][method_name] = [np.mean(roc_aucs),
                                                          0 if len(roc_aucs) == 1 else scipy.stats.sem(np.array(roc_aucs))
                                                          ]

    with open('results-{}.csv'.format(metric), 'w') as csvfile:
        fieldnames = ['dataset', 'single class name'] + sorted(list(methods))
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for ds_name in sorted(results.keys()):
            for sc_name in sorted(results[ds_name].keys()):
                row_dict = {'dataset': ds_name, 'single class name': sc_name}
                row_dict.update({method_name: '{:.3f} ({:.3f})'.format(*results[ds_name][sc_name][method_name])
                                 for method_name in results[ds_name][sc_name]})
                writer.writerow(row_dict)

def main():
    create_auc_table()

if __name__ == '__main__':
    main() 
    

