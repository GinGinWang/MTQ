import numpy as np
import torch
import torch.nn.functional as F

from models.base import BaseModule

# def kl_divergence_loss(self, mean, logvar):
#         klloss= ((mean**2 + logvar.exp() - 1 - logvar) / 2).sum() / mean.size(0)
#         self.klloss = klloss
#         return klloss
   

class AutoregressionLoss(BaseModule):
    """
    Implements the autoregression loss.
    Given a representation and the estimated cpds, provides
    the log-likelihood of the representation under the estimated prior.
    """
    def __init__(self):
        # type: (int) -> None
        """
        Class constructor.

        :param cpd_channels: number of bins in which the multinomial works.
        """
        super(AutoregressionLoss, self).__init__()

    def forward(self, mean, logvar ,size_average = True):
        klloss= ((mean**2 + logvar.exp() - 1 - logvar) / 2).sum() / mean.size(0)
        # if size_average:
        #     klloss = torch.mean(klloss)
        return klloss