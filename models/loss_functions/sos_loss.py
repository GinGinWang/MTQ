import numpy as np
import torch
import torch.nn.functional as F
from models.base import BaseModule
import math

#JJ: rewrite Autoregression Loss
class SoSLoss(BaseModule):
    """
    Implements the autoregression loss. Using sosflow (3)
    """
    def __init__(self):
        super(SoSLoss, self).__init__()

    def forward(self, s, log_jacob, use_J=True, size_average=True):
    # s = T^{-1}x
        log_probs = (-0.5 * s.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)
        # formula (3)
        # loss = -(log_probs + log_jacob).sum() if use_J else -(log_probs).sum()
        loss = -(log_probs + log_jacob).sum() if use_J else -(log_probs).sum()
        
        if size_average:
            loss /= s.size(0)
        return loss
