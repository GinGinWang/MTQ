import collections
import random
import re
from typing import List

import numpy as np
import torch


# def set_random_seed(seed):
#     # type: (int) -> None
#     """
#     Sets random seeds.
#     :param seed: the seed to be set for all libraries.
#     """
#     random.seed(seed)
#     np.random.seed(seed)
#     # np.random.shuffle.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


def normalize(samples, min, max):
    # type: (np.ndarray, float, float) -> np.ndarray
    """
    Normalize scores as in Eq. 10

    :param samples: the scores to be normalized.
    :param min: the minimum of the desired scores.
    :param max: the maximum of the desired scores.
    :return: the normalized scores
    """

    if (max - min) == 0:
         result = samples 
    else: 
         result =  (samples - min) / (max - min)

    # result =  (samples - min) / (max - min)
            
    return result


def novelty_score(sample_llk_norm, sample_rec_norm):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    """
    Computes the normalized novelty score given likelihood scores, reconstruction scores
    and normalization coefficients (Eq. 9-10).
    :param sample_llk_norm: array of (normalized) log-likelihood scores.
    :param sample_rec_norm: array of (normalized) reconstruction scores.
    :return: array of novelty scores.
    """

    # Sum
    ns = sample_llk_norm + sample_rec_norm

    return ns
