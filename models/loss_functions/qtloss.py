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
        z = z.detach()
        log_jacob = log_jacob.detach()
        
        m = normal.Normal(torch.tensor([1.0]).to('cuda'), torch.tensor([1.0]).to('cuda'))
        qz = m.icdf(z)
        qz = m.icdf(qz)
        qz = qz/2.0
        qz = qz.sum(dim = 1)
        
        
        log_jacob_d = log_jacob
                  
        # formula (3)
        loss = (- log_jacob_d).sum(-1,keepdim=True)

        
        if size_average:
            loss = loss.mean()
        else:
            loss = loss.squeeze(-1)
        return loss
