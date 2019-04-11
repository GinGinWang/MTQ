# tempory estimator 
# based on MAF

from typing import List

import torch
import torch.nn as nn

from models.base import BaseModule

from models.estimator_sos import EstimatorSoS
from models.estimator_maf import EstimatorMAF
from models.estimator_1D import Estimator1D

class DE(BaseModule):
    """
    Implements an estimator for 1-dimensional vectors.
    1-dimensional vectors arise from the encoding of images.
    This module is employed in MNIST and CIFAR10 LSA models.modul
    """

    def __init__(self, input_shape, num_blocks, est_name):

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
        super(DE, self).__init__()

        c, w, h = input_shape 

        self.code_length = c*w*h

        self.name = 'Estimator_'+est_name

        if est_name == 'MAF':
            self.T_inverse = EstimatorMAF(num_blocks, self.code_length) 

        elif est_name == 'SOS':
            self.T_inverse = EstimatorSOS(num_blocks, self.code_length)

        # 1-D estimator from LSA
        elif est_name == 'EN':
            self.model = Estimator1D(
                code_length=code_length,
                fm_list=[32, 32, 32, 32],
                cpd_channels=100
        )

    def forward(self, z):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of latent vectors.
        :return: the batch of tuples (s, q(s)) CPD estimates q(s).
        """
        h = z.view(-1, self.code_length)
        
        s,log_jacob_s = self.T_inverse(h)        
        
        return None, None, None,s,log_jacob_s
