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
    def __init__(self, input_shape, code_length):
        # type: (Tuple[int, int, int], int) -> None
        """
        Class constructor:

        :param input_shape: the shape of MNIST samples.
        :param code_length: the dimensionality of latent vectors.
        """
        super(Encoder, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length

        c, h, w = input_shape
        
        activation_fn = nn.LeakyReLU()

        # Convolutional network
        self.conv = nn.Sequential(
            DownsampleBlock(channel_in=c, channel_out=32, activation_fn=activation_fn),
            DownsampleBlock(channel_in=32, channel_out=64, activation_fn=activation_fn),
        )

        self.deepest_shape = (64, h // 4, w // 4)

        # FC network
        self.fc = nn.Sequential(
            nn.Linear(in_features=reduce(mul, self.deepest_shape), out_features=64),
            nn.BatchNorm1d(num_features=64),
            activation_fn,
            nn.Linear(in_features=64, out_features=code_length),
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
        h = self.conv(h)
        h = h.view(len(h), -1)
        o = self.fc(h)

        return o


class LSAET_MNIST(BaseModule):
    """
    LSA model for MNIST one-class classification.
    """
 

    def __init__(self,  input_shape, code_length, num_blocks, hidden_size, est_name = None):
        
        # type: (Tuple[int, int, int], int, int) -> None
        """
        Class constructor.

        :param input_shape: the shape of MNIST samples.
        :param code_length: the dimensionality of latent vectors.
        :param cpd_channels: number of bins in which the multinomial 
        works.
        :param est_name: density estimator {"sos","maf"}
        
        """
        super(LSAET_MNIST, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length
        self.est_name = est_name


        self.name = f'LSA_ET_{est_name}'

        # Build encoder
        self.encoder = Encoder(
            input_shape=input_shape,
            code_length=code_length
        )


        # Build estimator
        if est_name in ["SOS","QT"]:
            self.estimator = TinvSOS(num_blocks, code_length,hidden_size)      
        elif est_name == "MAF":
            self.estimator = TinvMAF(num_blocks, code_length,hidden_size)
        elif est_name == 'EN':
            self.estimator = Estimator1D(
            code_length=code_length,
            fm_list=[32, 32, 32, 32],
            cpd_channels=100)    
        else:
            ValueError('No estimator')
        print(f"{self.name} Model Initialization")
    def forward(self, x):
        # type: (torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        Forward propagation.

        :param x: the input batch of images.
        :return: a tuple of torch.Tensors holding reconstructions, latent vectors and CPD estimates.
        """

        # Produce representations
        z = self.encoder(x)
        
        
        if self.est_name == 'EN':
                # density distribution of z 
            z_dist = self.estimator(z)
        elif self.est_name in ['SOS','MAF']:
            s, log_jacob_T_inverse = self.estimator(z)

        

        # Without Estimator
        if self.est_name == 'EN':
            return z, z_dist
        elif self.est_name in ['MAF','SOS']:
                return s, log_jacob_T_inverse
        else:
            ValueError('No Estimator')
        