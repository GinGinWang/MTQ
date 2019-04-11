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
        super(SoSLoss, self).__init__()

    def forward(self, s, log_jacob, use_J=True, size_average=True):
        '''
        Args:
            s, source data s~ N(0,1) T(s) = z
            log_jacob: log of jacobian of T-inverse
            
        return: the mean negative log-likelihood (averaged along the batch axis)
        '''
        log_probs = (-0.5 * s.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)
        # formula (3)
        
        loss = -(log_probs + log_jacob).sum() if use_J else -(log_probs).sum()
        
        if size_average:
            loss /= s.size(0)

        return loss
