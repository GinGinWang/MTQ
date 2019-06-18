import numpy as np
import torch
import torch.nn.functional as F
from models.base import BaseModule
import math

import torch.distributions as normal

#JJ: rewrite Autoregression Loss
class QTLoss(BaseModule):
    """
    T(s) = z s~N(0,1), z~q
    loss = -mean( log q(z_i)), average loss in every batch
    """
    def __init__(self):
        super(QTLoss, self).__init__()

    def forward(self, z, log_jacob, size_average=True):
        '''
        Args:
            s, source data s~ N(0,1) T(s) = z
            log_jacob: log of jacobian of T-inverse

         
        return: the mean negative log-likelihood (averaged along the batch axis)
        '''
        # s_d = s.detach()
        # log_jacob_d = log_jacob.detach()
        # z_d = z.detach()
        # log_jacob_d = log_jacob.detach()
        
        z = z.view(-1,z.shape[1]*z.shape[2]*z.shape[3])
        print(z.max())
        print(z.min())
        
        
        m = normal.Normal(torch.zeros(z.shape[0],z.shape[1]).to('cuda'), torch.ones(z.shape[0],z.shape[1]).to('cuda'))
        qz = m.icdf(z)
        qz.clamp(max = 1, min = -1)
        # print(qz)
        # qz = qz.pow(2)
        # print(qz)
        # print(qz.shape)
        qz = qz.sum(dim = 1)
        # print(qz.shape)
        # print(qz)
        
        qz = qz / 2.0                  
        # formula (3)
        loss = (- log_jacob - qz).sum(-1,keepdim=True)

        
        if size_average:
            loss = loss.mean()
        else:
            loss = loss.squeeze(-1)
        return loss
