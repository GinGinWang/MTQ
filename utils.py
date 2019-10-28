"""

Utils: path,random seed, weight initialization.

By Jing Jing Wang/ 2019.10

"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import math


def create_checkpoints_dir(dataset, fixed, mulobj, num_blocks,
                           hidden_size, code_length, estimator):
    """
    Create the dir of checkpoint.

    for saving model or for loading model.
    """
    dirName = f'checkpoints/{dataset}/'
    if estimator in ['SOS', 'MAF']:
        dirName = f'{dirName}b{num_blocks}h{hidden_size}c{code_length}/'

    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print(f'Make Dir:{dirName}')
    return dirName


def create_file_path(mulobj, fixed, pretrained,
                     model_name, dataset, score_normed,
                     num_blocks, hidden_size, code_length, lam, checkpoint):
    """
    Create the  path for result table

    Used in Test mode.
    """
    result_name = f"{model_name}_{dataset}_nml{score_normed}_b{num_blocks}_h{hidden_size}_c{code_length}_lam{lam}"
    if mulobj:
        result_path = f"results/Mul_{result_name}"
    elif fixed:
        result_path = f"results/fix_{result_name}"
    elif pretrained:
        result_path = f"results/pre_{result_name}"
    else:
        result_path = f"results/{result_name}"

    result_path = f"{result_path}.txt"

    return result_path


def set_random_seed(seed):
    """
    Sets random seeds.

    :param seed: the seed to be set for all libraries.
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # np.random.shuffle.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _init_fn(worker_id):
    np.random.seed(int(seed))


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
