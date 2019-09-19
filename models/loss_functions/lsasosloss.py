import torch
import torch.nn as nn

from models.loss_functions.autoregression_loss import AutoregressionLoss
from models.loss_functions.reconstruction_loss import ReconstructionLoss

from models.loss_functions.flow_loss import FlowLoss

class LSASOSLoss(nn.Module):
    """
    Implements the loss of a LSA model.
    It is a sum of the reconstruction loss and the autoregression loss.
    """
    def __init__(self, lam=1):
        # type: (int, float) -> None
        """
        Class constructor.

        :param cpd_channels: number of bins in which the multinomial works.
        :param lam: weight of the autoregression loss.
        """
        super(LSASOSLoss, self).__init__()

        self.lam = lam

        # Set up loss modules
        self.reconstruction_loss_fn = ReconstructionLoss()
        self.autoregression_loss_fn = FlowLoss()

        # Numerical variables
        self.reconstruction_loss = None
        self.autoregression_loss = None

        self.total_loss = None
        self.nlog_probs = None
        self.nagtive_log_jacob = None

    def forward(self, x, x_r, s,nagtive_log_jacob, average = True):
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
        rec_loss = self.reconstruction_loss_fn(x, x_r, average)
        arg_loss, nlog_probs, nlog_jacob_d = self.autoregression_loss_fn(s,nagtive_log_jacob ,average)

        tot_loss = rec_loss + self.lam * arg_loss

        # Store numerical
        self.reconstruction_loss = rec_loss
        self.autoregression_loss = arg_loss
        
        self.nlog_probs = nlog_probs * self.lam
        self.nagtive_log_jacob = nlog_jacob_d * self.lam
        
        self.total_loss = tot_loss

        return tot_loss