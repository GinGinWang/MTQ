from typing import List

import torch
import torch.nn as nn

from models.base import BaseModule

##sos
import general_maf  as gmaf
import flows as fnn


class EstimatorSoS(BaseModule):
    """
    Implements an estimator for 1-dimensional vectors.
    1-dimensional vectors arise from the encoding of images.
    This module is employed in MNIST and CIFAR10 LSA models.modul

    Takes as input a latent vector z  and outputs (s,p(s))
    z= T(s), where T is sosflow
    p(s) = q(z), pd
    """

    def __init__(self, num_blocks, code_length, num_hidden=60, K=5, M=3, use_bn=True ):
        # type: (int, List[int], int) -> None
        """
        Class constructor.

        Args:
            num_blocks
            code_length
            num_hidden
            K
            M
            use_bn
        """
        super(EstimatorSoS, self).__init__()

        modules = []
        
        for _ in range(num_blocks):
            if use_bn:
                modules += [
                    gmaf.SumSqMAF(code_length, num_hidden, K, M),
                    fnn.BatchNormFlow(code_length),
                    fnn.Reverse(code_length)
                ]
            else:
                modules += [
                    gmaf.SumSqMAF(code_length, num_hidden, K, M),
                    fnn.Reverse(code_length)
                ]
        model = fnn.FlowSequential(*modules)

        # intialize
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                module.bias.data.fill_(0)

        self.T_inverse = model


    def forward(self, z):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of latent vectors.
        :return: the batch of tuples (s, q(s)) CPD estimates q(s).
        """
        s,log_jacob_s = self.T_inverse(z)

        return s,log_jacob_s