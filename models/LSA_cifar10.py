from functools import reduce
from operator import mul
from typing import Tuple

import torch
import torch.nn as nn

from models.base import BaseModule
from models.blocks_2d import DownsampleBlock
from models.blocks_2d import ResidualBlock
from models.blocks_2d import UpsampleBlock

from models.estimator_sos import EstimatorSoS
from models.estimator_maf import EstimatorMAF
import torch.nn.functional as F


class Encoder(BaseModule):
    """
    CIFAR10 model encoder.
    """
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


class LSA_CIFAR10(BaseModule):
    """
    LSA model for CIFAR10 one-class classification.
    """
    def __init__(self,  input_shape, code_length,num_blocks, est_name, combine_density):
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
       
        super(LSA_CIFAR10, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length
        self.est_name = est_name
        self.name = f"{est_name}LSA"

        # the input of estimator is density z / combine_density (z,|x-x_r|^2)
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

        # # Build estimator
        # self.estimator = Estimator1D(
        #     code_length=code_length,
        #     fm_list=[32, 32, 32, 32],
        #     cpd_channels=cpd_channels
        # )

        # Build estimator
        # Use New density estimator
            #sosflow
        if combine_density == False:
            if est_name == "SOS":
                self.estimator = EstimatorSoS(num_blocks, code_length)  
                # maf    
            elif est_name == "MAF":
                self.estimator = EstimatorMAF(num_blocks, code_length)
        else:

        # density = [z,dist(x,x_r)], code_length+1
            if est_name == "SOS":
                self.estimator = EstimatorSoS(num_blocks, code_length+1)  
                # maf    
            elif est_name == "MAF":
                self.estimator = EstimatorMAF(num_blocks, code_length+1)

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
        z_dist = None


        # Estimate CPDs with autoregression
        # density estimator
        if self.combine_density == False:
            s,log_jacob_s = self.estimator(z)
        
        # [z,|x-xr|^2]
        else:
            L = torch.pow((x - x_r), 2)

            while L.dim() > 1:
                L = torch.sum(L, dim=-1)
            
            L= torch.unsqueeze(L,1)
            new_z = torch.cat((z,L),1)

            s,log_jacob_s = self.estimator(new_z)

        return x_r, z, z_dist,s,log_jacob_s
