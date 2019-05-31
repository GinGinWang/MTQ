from functools import reduce
from operator import mul
from typing import Tuple

import torch
import torch.nn as nn


# LSA
from models.base import BaseModule
from models.blocks_2d import DownsampleBlock
from models.blocks_2d import UpsampleBlock 


# flows
from models.transform_sos import TinvSOS
from models.transform_maf import TinvMAF

# estimator
from models.estimator_1D import Estimator1D


import torch.nn.functional as F


class Encoder(BaseModule):
    """
    MNIST model encoder.
    same as LSA
    """
    def __init__(self):
        # type: (Tuple[int, int, int], int) -> None
        """
        Class constructor:

        :param input_shape: the shape of MNIST samples.
        :param code_length: the dimensionality of latent vectors.
        """
        super(Encoder, self).__init__()

        
        activation_fn = nn.Hardtanh()

        self.fc = nn.Sequential(
            nn.Linear(in_features=121, out_features=60),
            # nn.BatchNorm1d(num_features=60),
            activation_fn,
            nn.Linear(in_features=60, out_features=30),
            activation_fn,
            nn.Linear(in_features=30, out_features=10),
            activation_fn,
            nn.Linear(in_features=10, out_features=5),
            nn.Sigmoid()
        )

    def forward(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the input batch of images.
        :return: the batch of latent vectors.
        """

        h = x  
        # h = self.conv(h)
        h = h.view(len(h), -1)
        o = self.fc(h)

        return o


class Decoder(BaseModule):
    """
    MNIST model decoder.
    """
    def __init__(self):
        # type: (int, Tuple[int, int, int], Tuple[int, int, int]) -> None
        """
        Class constructor.
        same as LSA

        :param code_length: the dimensionality of latent vectors.
        :param deepest_shape: the dimensionality of the encoder's deepest convolutional map.
        :param output_shape: the shape of MNIST samples.
        """
        super(Decoder, self).__init__()

        activation_fn = nn.Hardtanh()

        # FC network
        self.fc = nn.Sequential(
            nn.Linear(in_features=5, out_features=10),
            # nn.BatchNorm1d(num_features=64),
            activation_fn,
            nn.Linear(in_features=10, out_features=30),
            activation_fn,
            nn.Linear(in_features=30, out_features=60),
            activation_fn,
            nn.Linear(in_features=60, out_features=121),
            # nn.BatchNorm1d(num_features=self.output_shape),
            activation_fn
        )




    def forward(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of latent vectors.
        :return: the batch of reconstructions.
        """
        h = x.view(len(x),-1)
        h = self.fc(x)
        o = h
        return o


class LSA_KDDCUP(BaseModule):
    """
    LSA model for MNIST one-class classification.
    """
 

    def __init__(self,num_blocks,hidden_size):
        # type: (Tuple[int, int, int], int, int) -> None
        """
        Class constructor.

        :param input_shape: the shape of MNIST samples.
        :param code_length: the dimensionality of latent vectors.
        :param cpd_channels: number of bins in which the multinomial 
        works.
        :param est_name: density estimator {"sos","maf"}
        :param combine_density: 
                            False =  input of estimator is z
                            True  =  input of estimator is (z,|x-xr|^2)

        """
        super(LSA_KDDCUP, self).__init__()

        
        self.coder_name = 'LSA'

        self.name = f'LSA_SOS'
        
        # Build encoder
        self.encoder = Encoder(
             )

        # Build decoder
        self.decoder = Decoder(
        )

        self.estimator = TinvSOS(num_blocks, 5, hidden_size)      
            

    def forward(self, x):
        # type: (torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        Forward propagation.

        :param x: the input batch of images.
        :return: a tuple of torch.Tensors holding reconstructions, latent vectors and CPD estimates.
        """

        # Produce representations
        z = self.encoder(x)
        # Reconstruct x
        x_r = self.decoder(z)
        x_r = x_r.view(-1,*x.shape)
        
        # Estimate CPDs with autoregression
        # density estimator
        s, log_jacob_T_inverse = self.estimator(z)

        return  x_r, z, s, log_jacob_T_inverse
        