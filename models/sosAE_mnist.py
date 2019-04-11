'''
SosAE = VanillaAE with SoSflow  
'''
from functools import reduce
from operator import mul
from typing import Tuple

import torch
import torch.nn as nn

from models.base import BaseModule
from models.blocks_2d import DownsampleBlock
from models.blocks_2d import UpsampleBlock
#JJ: replace density estimator 
from models.estimator_sos import EstimatorSoS
import torch.nn.functional as F

class Encoder(BaseModule):
    """
    MNIST model encoder.
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

  
        d = 128
        self.conv1_1 = nn.Conv2d(c, d//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d // 2, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, code_length, 4, 1, 0)

    def forward(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the input batch of images.
        :return: the batch of latent vectors.
        """
        h = F.leaky_relu(self.conv1_1(x), 0.2)
        h = F.leaky_relu(self.conv2_bn(self.conv2(h)), 0.2)
        h = F.leaky_relu(self.conv3_bn(self.conv3(h)), 0.2)
        o = self.conv4(h)

        return o


class Decoder(BaseModule):
    """
    MNIST model decoder.
    """
    def __init__(self, code_length, output_shape):
        # type: (int, Tuple[int, int, int], Tuple[int, int, int]) -> None
        """
        Class constructor.

        :param code_length: the dimensionality of latent vectors.
        :param output_shape: the shape of MNIST samples.
        """
        super(Decoder, self).__init__()

        self.code_length = code_length
        # self.deepest_shape = deepest_shape
        self.output_shape = output_shape

        c, h, w = output_shape

        d = 128
        self.deconv1_1 = nn.ConvTranspose2d(code_length, d*2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)
        self.deconv1_2 = nn.ConvTranspose2d(10, d*2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d*2)
        self.deconv2 = nn.ConvTranspose2d(d*2, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, c, 4, 2, 1)

    def forward(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of latent vectors.
        :return: the batch of reconstructions.
        """
        h = F.relu(self.deconv1_1_bn(self.deconv1_1(x)))
        h = F.relu(self.deconv2_bn(self.deconv2(h)))
        h = F.relu(self.deconv3_bn(self.deconv3(h)))
        o = F.tanh(self.deconv4(h)) * 0.5 + 0.5
        return o


class SOSAE_MNIST(BaseModule):
    """
    LSA model for MNIST one-class classification.
    """
    def __init__(self,  input_shape, code_length, num_blocks):
        # type: (Tuple[int, int, int], int, int) -> None
        """
        Class constructor.

        :param input_shape: the shape of MNIST samples.
        :param code_length: the dimensionality of latent vectors.
        :param num_blocks: number of SoSflow blocks.
        """
        super(SOSAE_MNIST, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length
        self.num_blocks = num_blocks

        # Build encoder
        self.encoder = Encoder(
            input_shape=input_shape,
            code_length=code_length
        )

        # Build decoder
        self.decoder = Decoder(
            code_length=code_length,
            output_shape=input_shape
        )

        # Build estimator
        self.estimator = EstimatorSoS(num_blocks, code_length)

    def forward(self, x):
        # type: (torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]
        """
        Forward propagation.

        :param x: the input batch of images.
        :return: a tuple of torch.Tensors holding reconstructions, latent vectors  s and log_jacob_s= log|T'(s)|.
        """
        h = x

        # Produce representations
        z = self.encoder(h)

        # density estimator
        s,log_jacob_s = self.estimator(z)

        # Reconstruct x
        x_r = self.decoder(z)
        x_r = x_r.view(-1, *self.input_shape)

        return x_r, z, s, log_jacob_s
