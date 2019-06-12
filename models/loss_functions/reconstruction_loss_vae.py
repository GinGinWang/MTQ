import torch

from models.base import BaseModule
import torch.nn as nn


    # def reconstruction_loss(self, x_reconstructed, x):

    #     recloss = nn.BCELoss(size_average=False)(x_reconstructed, x) / x.size(0)
    #     self.recloss = recloss
    #     return recloss 


class ReconstructionLoss(BaseModule):
    """
    Implements the reconstruction loss.
    """
    def __init__(self):
        # type: () -> None
        """
        Class constructor.
        """
        super(ReconstructionLoss, self).__init__()

    def forward(self, x, x_r, average = True):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of input samples.
        :param x_r: the batch of reconstructions.
        :return: the mean reconstruction loss (averaged along the batch axis).
        """

        L = nn.BCELoss(size_average=False)(x_r, x) / x.size(0)

        # while L.dim() > 1:
        #     L = torch.sum(L, dim=-1)
        # if average: # whether got average in batch
        #     L = torch.mean(L)
        return L