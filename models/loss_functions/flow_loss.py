import numpy as np
import torch
import torch.nn.functional as F
from models.base import BaseModule
import math

#JJ: rewrite Autoregression Loss
class FlowLoss(BaseModule):
    """
    T(s) = z s~N(0,1), z~q
    loss = -mean( log q(z_i)), average loss in every batch
    """
    def __init__(self):
        super(FlowLoss, self).__init__()

    def forward(self, s, log_jacob,size_average=True):
        '''
        Args:
            s, source data s~ N(0,1) T(s) = z
            log_jacob: log of jacobian of T-inverse

         
        return: the mean negative log-likelihood (averaged along the batch axis)
        '''
        s_d = s
        log_jacob_d = log_jacob
        
        log_probs = (-0.5 * s_d.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)
        
        # formula (3)
        loss = -(log_probs + log_jacob_d).sum(-1, keepdim=True)
        log_probs = log_probs.sum(-1,keepdim=True)
        log_jacob_d = log_jacob_d.sum(-1,keepdim=True)

        if size_average:
            loss = loss.mean()
            log_probs= log_probs.mean()
            log_jacob_d = log_jacob_d.mean()
        else:
            loss = loss.squeeze(-1)
            log_probs= log_probs.squeeze(-1)
            log_jacob_d = log_jacob_d.squeeze(-1)

        return loss, -log_probs,-log_jacob_d
