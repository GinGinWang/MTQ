import torch
import torch.nn as nn

from models.loss_functions.autoregression_loss_vae import AutoregressionLoss
from models.loss_functions.reconstruction_loss_vae import ReconstructionLoss


class VAELoss(nn.Module):
    """
    Implements the loss of a LSA model.
    It is a sum of the reconstruction loss and the autoregression loss.
    """
    def __init__(self):
        # type: (int, float) -> None
        """
        Class constructor.

        :param cpd_channels: number of bins in which the multinomial works.
        :param lam: weight of the autoregression loss.
        """
        super(VAELoss, self).__init__()

        # Set up loss modules
        self.reconstruction_loss_fn = ReconstructionLoss()
        self.autoregression_loss_fn = AutoregressionLoss()

        self.reconstruction_loss = None
        self.autoregression_loss = None
        self.tot_loss = None
        # Numerical variables
        
    def forward(self, x, x_r, mean, logvar, average=True):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of input samples.
        :param x_r: the batch of reconstructions.
        :param z: the batch of latent representations.
        :param z_dist: the batch of estimated cpds.
        :return: the loss of the model (averaged along the batch axis).
        """
        # Compute pytorch loss
        rec_loss = self.reconstruction_loss_fn(x, x_r)
        llk_loss = self.autoregression_loss_fn(mean,logvar)

        tot_loss = rec_loss+ llk_loss

        # Store numerical
        self.reconstruction_loss = rec_loss
        self.autoregression_loss = llk_loss

        self.total_loss = tot_loss

        return tot_loss