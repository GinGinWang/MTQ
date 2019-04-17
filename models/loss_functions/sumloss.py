import torch
import torch.nn as nn


from models.loss_functions.autoregression_loss import AutoregressionLoss
from models.loss_functions.reconstruction_loss import ReconstructionLoss

from models.loss_functions.flow_loss import FlowLoss

class SumLoss(nn.Module):
    """
    Implements the loss of a LSA model.
    It is a sum of the reconstruction loss and the autoregression loss.
    """

    def __init__(self, model_name, cpd_channels=100, lam=1):

        # type: (int, float) -> None
        """
        Class constructor.

        :param cpd_channels: number of bins in which the multinomial works.
        :param lam: weight of the autoregression loss.
        """
        super(SumLoss, self).__init__()

        self.cpd_channels = cpd_channels
        self.lam = lam

        self.name = model_name

        # Set up loss modules
        self.reconstruction_loss_fn = ReconstructionLoss()
        
        self.autoregression_loss_fn = AutoregressionLoss(self.cpd_channels)

        self.flow_loss_fn = FlowLoss()

        
        # Reconstruction Loss
        self.reconstruction_loss = None
        # Negative Log-likelihood of latent vector z 
        self.nllk = None

        # Add all needed loss
        self.total_loss = None


    def lsa(self, x,x_r, batch_average= True):
        self.reconstruction_loss = self.reconstruction_loss_fn( x,  x_r, batch_average )
        self.total_loss = self.reconstruction_loss

    def lsa_en(self, x, x_r, z, z_dist, batch_average= True):
        
        self.reconstruction_loss = self.reconstruction_loss_fn(x, x_r,batch_average)

        self.nllk = self.autoregression_loss_fn(z,z_dist,batch_average)
        
        self.total_loss = self.reconstruction_loss + self.lam *self.nllk

    def lsa_flow(self, x, x_r, s,nagtive_log_jacob, batch_average= True):
        self.reconstruction_loss = self.reconstruction_loss_fn(x, x_r,batch_average)
        self.nllk= self.flow_loss_fn(s,nagtive_log_jacob,batch_average)
        self.total_loss = self.lam * self.nllk + self.reconstruction_loss

    def flow(self, s,nagtive_log_jacob, batch_average= True):
        self.nllk = self.flow_loss_fn(s,nagtive_log_jacob,batch_average)
        self.total_loss = self.nllk
    
    def en(self, z_dist, batch_average= True):
        self.nllk = self.autoregression_loss_fn(z_dist,batch_average)
        self.total_loss = self.nllk