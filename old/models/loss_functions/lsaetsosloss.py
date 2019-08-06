import torch
import torch.nn as nn

from models.loss_functions.autoregression_loss import AutoregressionLoss


from models.loss_functions.flow_loss import FlowLoss

class LSAETSOSLoss(nn.Module):
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
        super(LSAETSOSLoss, self).__init__()

        # Set up loss modules
        self.autoregression_loss_fn = FlowLoss()

        # Numerical variables
        self.autoregression_loss = None
        self.total_loss = None

    def forward(self, s,nagtive_log_jacob,average = True):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param z: the batch of latent representations.
        :param z_dist: the batch of estimated cpds.
        :return: the loss of the model (averaged along the batch axis).
        """
        # Compute pytorch loss
        arg_loss = self.autoregression_loss_fn(s, nagtive_log_jacob,average)
        tot_loss =  arg_loss

        # Store numerical
        self.autoregression_loss = arg_loss
        self.total_loss = tot_loss

        return tot_loss