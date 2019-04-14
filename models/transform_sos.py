from typing import List

import torch
import torch.nn as nn

from models.base import BaseModule

from flow_sos_models import *


class TinvSOS(BaseModule):
    
    """
    Implements an sos-flow for 1-dimensional vectors.
    1-dimensional vectors arise from the encoding of images.

    Model: T-inverse , T-inverse(z) = s, where T-inverse is built by SoSflow

        Input: latent vector z 
        Output: s, -log_jacob of T (i.e., logjab of T-inverse)
    """

    def __init__(self, n_blocks, input_size, hidden_size=1024, k=5, r=3, device = None,**kwargs):

        # type: (int, List[int], int) -> None
        """
        Class constructor: build SoS-flow

        Args:
            num_blocks
            input_size: dimension of input data z  
            num_hidden: neuron number in every hidden layer
            K: number of squares to be summed up
            M: degree of polynomial in every square part
           """

        super(TinvSOS, self).__init__()
        self.name = 'SOS'
        self.input_size = input_size
        modules = []
        for i in range(n_blocks):
            modules += [
                    SOSFlow(input_size, hidden_size, k, r),
                    BatchNormFlow(input_size),
                    Reverse(input_size)
                ]
        
        model = FlowSequential(*modules)
        if device is not None:
            model.to(device)

        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)

        self.T_inverse = model


    def forward(self, z):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param z: the batch of latent vectors.
        :return: s= T_inverse(z), log_jacob_T_inv.
        """
        h = z.view(-1, self.input_size)
        s,log_jacob_T_inv = self.T_inverse(h)

        return s,log_jacob_T_inv