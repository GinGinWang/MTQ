# tempory estimator 
# based on MAF

from typing import List

import torch
import torch.nn as nn

from models.base import BaseModule

##sos
import general_maf  as gmaf
# 
# import flows as fnn
import flows_maf as fnn

class EstimatorMAF(BaseModule):
    """
    Implements an estimator for 1-dimensional vectors.
    1-dimensional vectors arise from the encoding of images.
    This module is employed in MNIST and CIFAR10 LSA models.modul

    Takes as input a latent vector z  and outputs (s,p(s))
    z= T(s), where T is sosflow
    p(s) = q(z), pd
    """

    def __init__(self, num_blocks, code_length, num_hidden=60, use_bn=True ):
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
        num_inputs = code_length
        num_cond_inputs = None
        act = 'relu'


        super(EstimatorMAF, self).__init__()

        code_length = code_length
        modules = []
        for _ in range(num_blocks):
            modules += [
                fnn.MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
                fnn.BatchNormFlow(num_inputs),
                fnn.Reverse(num_inputs)
            ]

        model = fnn.FlowSequential(*modules)

        # intialize
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
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