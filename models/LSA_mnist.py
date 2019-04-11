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


class Decoder(BaseModule):
    """
    MNIST model decoder.
    """
    def __init__(self, code_length, deepest_shape, output_shape):
        # type: (int, Tuple[int, int, int], Tuple[int, int, int]) -> None
        """
        Class constructor.
        same as LSA

        :param code_length: the dimensionality of latent vectors.
        :param deepest_shape: the dimensionality of the encoder's deepest convolutional map.
        :param output_shape: the shape of MNIST samples.
        """
        super(Decoder, self).__init__()

        self.code_length = code_length
        self.deepest_shape = deepest_shape
        self.output_shape = output_shape

        activation_fn = nn.LeakyReLU()

        # FC network
        self.fc = nn.Sequential(
            nn.Linear(in_features=code_length, out_features=64),
            nn.BatchNorm1d(num_features=64),
            activation_fn,
            nn.Linear(in_features=64, out_features=reduce(mul, deepest_shape)),
            nn.BatchNorm1d(num_features=reduce(mul, deepest_shape)),
            activation_fn
        )

        # Convolutional network
        self.conv = nn.Sequential(
            UpsampleBlock(channel_in=64, channel_out=32, activation_fn=activation_fn),
            UpsampleBlock(channel_in=32, channel_out=16, activation_fn=activation_fn),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=False)
        )



    def forward(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of latent vectors.
        :return: the batch of reconstructions.
        """
        h = x
        h = self.fc(h)
        h = h.view(len(h), *self.deepest_shape)
        h = self.conv(h)
        o = h
        return o


class LSA_MNIST(BaseModule):
    """
    LSA model for MNIST one-class classification.
    """
 

    def __init__(self,  input_shape, code_length, num_blocks, est_name = None, combine_density= False):
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
        super(LSA_MNIST, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length
        self.est_name = est_name

        self.coder_name = 'LSA'

        if est_name == None:  
            self.name = f'LSA_{est_name}'
        else:
            self.name = 'LSA'
        # the input of estimator is latent vector z / combine_latentvector (z,|x-x_r|^2)

        self.combine_density = combine_density

        # Build encoder
        self.encoder = Encoder(
            input_shape=input_shape,
            code_length=code_length
        )

        # Build decoder
        self.decoder = Decoder(
            code_length=code_length,
            deepest_shape=self.encoder.deepest_shape,
            output_shape=input_shape
        )

        # Build estimator
        if combine_density:
            # sos- flow : T-inverse(z) = s,
            # output: s, -log_jacobian 
            if est_name == 'SOS':
                self.estimator = TinvSOS(num_blocks, code_length+1)  
            
            # maf- flow : T-inverse(z) = s,
            # output: s, -log_jacobian 
            elif est_name == 'MAF':
                self.estimator = TinvMAF(num_blocks, code_length+1)
            
            # estimation network: T(z)= p(z)
            # output p(z)
            elif est_name == 'EN':
                self.estimator = Estimator1D(
                code_length=code_length+1,
                fm_list=[32, 32, 32, 32],
                cpd_channels=100)
            # No estimator
            elif est_name == None:
                self.estimator = None


        else:
            if est_name == "SOS":
                self.estimator = TinvSOS(num_blocks, code_length)      
            elif est_name == "MAF":
                self.estimator = TinvMAF(num_blocks, code_length)
            elif est_name == 'EN':
                self.estimator = Estimator1D(
                code_length=code_length,
                fm_list=[32, 32, 32, 32],
                cpd_channels=100
        )   
            # No estimator
            elif est_name == None:
                self.estimator = None



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
        x_r = x_r.view(-1, *self.input_shape)
        



        # Estimate CPDs with autoregression
        # density estimator

        if self.combine_density:
            
            # whether need normalize?
            L = torch.pow((x - x_r), 2)
            while L.dim() > 1:
                 L = torch.sum(L, dim=-1)
            # L = L.view(-1,len(z))
            L.unsqueeze_(-1)     
            new_z = torch.cat((z,L),1)
            if est_name == 'EN':
                # density distribution of z 
                z_dist= self.estimator(new_z)
            elif est_name == 'SOS' or est_name =='MAF':
                s, log_jacob_T_inverse = self.estimator(new_z)
        else:
            if est_name == 'EN':
                # density distribution of z 
                z_dist= self.estimator(z)
            elif est_name == 'SOS' or est_name =='MAF':
                s, log_jacob_T_inverse = self.estimator(z)

        # Without Estimator
        if self.estimator == None:
            return x_r
        elif self.estimator == 'EN':
            return x_r, z, z_dist
        elif self.estimator == 'MAF' or self.estimator == 'SOS':
            return x_r, z, s, log_jacob_T_inverse
