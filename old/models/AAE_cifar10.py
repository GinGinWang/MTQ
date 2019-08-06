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


class Encoder(nn.Module):
    # initializers
    def __init__(self, input_shape, code_length):  
        super(Encoder, self).__init__()
        d =128
        self.input_shape = input_shape
        self.code_length = code_length

        c, h, w = input_shape

        self.conv1_1 = nn.Conv2d(c, d//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d // 2, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, code_length, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = self.conv4(x)
        return x


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





class AAE_CIFAR10(BaseModule):
    """
    AAE model for MNIST one-class classification.
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
        super(AAE_CIFAR10, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length
        self.est_name = est_name

        self.coder_name = 'AAE'

        if est_name == None:  
            self.name = 'AAE'
            print(f'{self.name}_cd_{combine_density}')
        else:
            self.name = f'AAE_{est_name}'
            print(f'{self.name}_cd_{combine_density}')
        
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
            output_shape=input_shape
        )

        # Build estimator
        if combine_density:
            # sos- flow : T-inverse(z) = s,
            # output: s, -log_jacobian 
            if est_name == 'SOS':
                self.estimator = TinvSOS(num_blocks, code_length+1, hidden_size)  
            
            # maf- flow : T-inverse(z) = s,
            # output: s, -log_jacobian 
            elif est_name == 'MAF':
                self.estimator = TinvMAF(num_blocks, code_length+1,hidden_size)
            
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
                self.estimator = TinvSOS(num_blocks, code_length,hidden_size)      
            elif est_name == "MAF":
                self.estimator = TinvMAF(num_blocks, code_length,hidden_size)
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
            # L = torch.pow((x-x_r),2)
            # normalized version is a little better

            L = torch.pow((x - x_r), 2)
            Lxr = torch.pow(x_r,2)
            
            while L.dim() > 1:
                 L = torch.sum(L, dim=-1)
                 Lxr = torch.sum(Lxr,dim=-1)

            # L = L.view(-1,len(z))

            L.unsqueeze_(-1)     
            Lxr.unsqueeze_(-1)
            L = L/ Lxr   
            z = torch.cat((z,L),1)

        if self.est_name == 'EN':
                # density distribution of z 
            z_dist= self.estimator(z)
        elif self.est_name in ['SOS','MAF']:
            s, log_jacob_T_inverse = self.estimator(z)

        # Without Estimator
        if self.est_name == 'EN':
            return x_r, z, z_dist
        elif self.est_name in ['MAF','SOS']:
            return x_r, z, s, log_jacob_T_inverse
        else:
            return x_r