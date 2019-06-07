from functools import reduce
from operator import mul
from typing import Tuple

import torch
import torch.nn as nn

from models.base import BaseModule
from models.blocks_2d import DownsampleBlock
from models.blocks_2d import ResidualBlock
from models.blocks_2d import UpsampleBlock

# flows
from models.transform_sos import TinvSOS
from models.transform_maf import TinvMAF

# estimator
from models.estimator_1D import Estimator1D

import torch.nn.functional as F


class Encoder(BaseModule):
   

    def __init__(self, input_shape, code_length):
        # type: (Tuple[int, int, int], int) -> None
        """
        Class constructor:

        :param input_shape: the shape of CIFAR10 samples.
        :param code_length: the dimensionality of latent vectors.
        """
        super(Encoder, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length

        c, h, w = input_shape
        
        activation_fn = nn.LeakyReLU()

        # Convolutional network
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, bias=False),
            activation_fn,
            ResidualBlock(channel_in=32, channel_out=32, activation_fn=activation_fn),
            DownsampleBlock(channel_in=32, channel_out=64, activation_fn=activation_fn),
            DownsampleBlock(channel_in=64, channel_out=128, activation_fn=activation_fn),
            DownsampleBlock(channel_in=128, channel_out=256, activation_fn=activation_fn),
        )
        self.deepest_shape = (256, h // 8, w // 8)

        # FC network
        self.fc = nn.Sequential(
            nn.Linear(in_features=reduce(mul, self.deepest_shape), out_features=256),
            nn.BatchNorm1d(num_features=256),
            activation_fn,
            nn.Linear(in_features=256, out_features=code_length),
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
    CIFAR10 model decoder.
    """
    def __init__(self, code_length, deepest_shape, output_shape):
        # type: (int, Tuple[int, int, int], Tuple[int, int, int]) -> None
        """
        Class constructor.

        :param code_length: the dimensionality of latent vectors.
        :param deepest_shape: the dimensionality of the encoder's deepest convolutional map.
        :param output_shape: the shape of CIFAR10 samples.
        """
        super(Decoder, self).__init__()

        self.code_length = code_length
        self.deepest_shape = deepest_shape
        self.output_shape = output_shape

        activation_fn = nn.LeakyReLU()

        # FC network
        self.fc = nn.Sequential(
            nn.Linear(in_features=code_length, out_features=256),
            nn.BatchNorm1d(num_features=256),
            activation_fn,
            nn.Linear(in_features=256, out_features=reduce(mul, deepest_shape)),
            nn.BatchNorm1d(num_features=reduce(mul, deepest_shape)),
            activation_fn
        )

        # Convolutional network
        self.conv = nn.Sequential(
            UpsampleBlock(channel_in=256, channel_out=128, activation_fn=activation_fn),
            UpsampleBlock(channel_in=128, channel_out=64, activation_fn=activation_fn),
            UpsampleBlock(channel_in=64, channel_out=32, activation_fn=activation_fn),
            ResidualBlock(channel_in=32, channel_out=32, activation_fn=activation_fn),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, bias=False)
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


class VAE(BaseModule):
    """
    LSA model for MNIST one-class classification.
    """
 

    def __init__(self,  input_shape, code_length, num_blocks, hidden_size, est_name = None, combine_density= False):
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
        super(VAE, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length
        self.est_name = est_name

        self.coder_name = 'LSA'

        if est_name == None:  
            self.name = 'LSA'
            print(f'{self.name}_cd_{combine_density}')
        else:
            self.name = f'LSA_{est_name}'
            print(f'{self.name}_cd_{combine_density}')
        
        # the input of estimator is latent vector z / combine_latentvector (z,|x-x_r|^2)

        
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



    def forward(self, x):
        # type: (torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        
        # encode x
        encoded = self.encoder(x)

        # sample latent code z from q given x.
        mean, logvar = self.q(encoded)
        z = self.z(mean, logvar)
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )

        # reconstruct x from z
        x_reconstructed = self.decoder(z_projected)

        # return the parameters of distribution of q given x and the
        # reconstructed image.
        return (mean, logvar), x_reconstructed

