import torch
import torch.nn as nn

<<<<<<< e68b04d9643bf8aa75b53953df98d650ab4d948c
from models.loss_functions.autoregression_loss import AutoregressionLoss
from models.loss_functions.reconstruction_loss import ReconstructionLoss

from models.loss_functions.sos_loss import SoSLoss
=======
from models.loss_functions.lsa_autoregression_loss import AutoregressionLoss
from models.loss_functions.reconstruction_loss import ReconstructionLoss

from models.loss_functions.flow_loss import FlowLoss
>>>>>>> message

class SumLoss(nn.Module):
    """
    Implements the loss of a LSA model.
    It is a sum of the reconstruction loss and the autoregression loss.
    """
<<<<<<< e68b04d9643bf8aa75b53953df98d650ab4d948c
    def __init__(self, lossname, cpd_channels=100, lam=1):
=======
    def __init__(self, model_name, cpd_channels=100, lam=1):
>>>>>>> message
        # type: (int, float) -> None
        """
        Class constructor.

        :param cpd_channels: number of bins in which the multinomial works.
        :param lam: weight of the autoregression loss.
        """
        super(SumLoss, self).__init__()

        self.cpd_channels = cpd_channels
        self.lam = lam
<<<<<<< e68b04d9643bf8aa75b53953df98d650ab4d948c
        self.lossname =lossname

        # Set up loss modules
        self.reconstruction_loss_fn = ReconstructionLoss()
        self.autoregression_loss_fn = AutoregressionLoss(self.cpd_channels)
        self.sos_loss_fn = SoSLoss()

        # Numerical variables
        self.reconstruction_loss = None
        self.autoregression_loss = None
=======

        self.name = model_name

        # Set up loss modules
        self.reconstruction_loss_fn = ReconstructionLoss()
        
        self.autoregression_loss_fn = AutoregressionLoss(self.cpd_channels)

        self.flow_loss_fn = FlowLoss()

        # Reconstruction Loss
        self.reconstruction_loss = None
        # Negative Log-likelihood of latent vector z 
        self.nnlk = None
>>>>>>> message

        # Add all needed loss
        self.total_loss = None

<<<<<<< e68b04d9643bf8aa75b53953df98d650ab4d948c
    def forward(self, x, x_r, z, z_dist, s, log_jacob_s):
=======
    # def forward(self, x, x_r, z, z_dist, s, log_jacob_s):
>>>>>>> message
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of input samples.
        :param x_r: the batch of reconstructions.
        :param z: the batch of latent representations.
        :param z_dist: the batch of estimated cpds.
        :
        :
        :return: the loss of the model (averaged along the batch axis).
        """
<<<<<<< e68b04d9643bf8aa75b53953df98d650ab4d948c
        # Compute pytorch loss
        if self.lossname == "LSA":
            self.autoregression_loss = self.autoregression_loss_fn(z, z_dist)
        if self.lossname == "SOSLSA":
            # rec_loss = self.reconstruction_loss_fn(x, x_r)
            self.autoregression_loss = self.sos_loss_fn(s,log_jacob_s)

        if self.lossname == "MAFLSA":
            # rec_loss = self.reconstruction_loss_fn(x, x_r)
            self.autoregression_loss = self.sos_loss_fn(s,log_jacob_s)

        # Store numerical
        self.reconstruction_loss = self.reconstruction_loss_fn(x,x_r)

        self.total_loss = self.reconstruction_loss+self.lam *self.autoregression_loss
        
=======
#         # Compute pytorch loss
#         if self.name == 'LSA':
#             self.autoregression_loss = self.autoregression_loss_fn(z, z_dist)
#             # Store numerical
#             self.reconstruction_loss = self.reconstruction_loss_fn(x,x_r)

#         if self.name == "SOSLSA":
#             # rec_loss = self.reconstruction_loss_fn(x, x_r)
#             self.autoregression_loss = self.sos_loss_fn(s,log_jacob_s)
# # Store numerical
#             self.reconstruction_loss = self.reconstruction_loss_fn(x,x_r)

#         if self.name == "MAFLSA":
#             # rec_loss = self.reconstruction_loss_fn(x, x_r)
#             self.autoregression_loss = self.autoregression_loss(s,log_jacob_s)
#             # Store numerical
#             self.reconstruction_loss = self.reconstruction_loss_fn(x,x_r)
#         if self.name == 'DEMAF':
#             self.autoregression_loss = self.autoregression_loss(s,log_jacob_s)
#             self.reconstruction_loss = 0
        
#         self.total_loss = self.reconstruction_loss+self.lam *self.autoregression_loss
    
    def lsa(x,x_r):
        self.reconstruction_loss = self.reconstruction_loss_fn( x, x_r)
        self.total_loss = self.reconstruction_loss

    def lsa_en(x,x_r,z,z_dist):
        self.reconstruction_loss = self.reconstruction_loss_fn(x, x_r)
        self.nnlk = self.autoregression_loss_fn(z,z_dist)
        self.total_loss = self.reconstruction_loss + self.nnlk

    def lsa_flow(x,x_r,s,nagtive_log_jacob):
        self.reconstruction_loss = self.reconstruction_loss_fn(x, x_r)
        self.nnlk= self.flow_loss_fn(s,nagtive_log_jacob)
        self.total_loss = self.nnlk + self.reconstruction_loss

    def flow(s,nagtive_log_jacob):
        self.nnlk = self.flow_loss_fn(s,nagtive_log_jacob)
        self.total_loss = self.nnlk
    
    def en(z_dist):
        self.nllk = self.autoregression_loss_fn(z_dist)
        self.total_loss = self.nnlk
>>>>>>> message
