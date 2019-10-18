"""
This is sos flow.

Model from sosflow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output.

    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class MaskedLinear(nn.Linear):
    """
    MaskedLinear.

    From.
    """

    def __init__(self, in_features, out_features, mask, bias=True):
        """
        Intialization.

        From
        """
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', mask)

    def forward(self, inputs):
        """
        Forward.

        From
        """
        return F.linear(inputs, self.weight * self.mask, self.bias)


class ConditionerNet(nn.Module):
    """
    Conditioner Net.

    From
    """

    def __init__(self, input_size, hidden_size, k, m, n_layers=1):
        """
        Init.

        From
        """
        super().__init__()
        # k: number of squares
        self.k = k
        # m = r+1, r: degree of polynomials in every squares
        self.m = m
        # input_size: dimension of input data
        self.input_size = input_size
        # output_size: number of parameters of conditioner
        self.output_size = k * self.m * input_size + input_size
        self.network = self._make_net(input_size, hidden_size,
                                      self.output_size, n_layers)

    def _make_net(self, input_size, hidden_size, output_size, n_layers):

        # n_layers is fixed to 1 here
        if self.input_size > 1:
            # make sure the the i_th output of net only dependent
            # on 1,..i-1 th input of net

            input_mask = get_mask(
                input_size, hidden_size, input_size, mask_type='input')
            hidden_mask = get_mask(hidden_size, hidden_size, input_size)
            output_mask = get_mask(
                hidden_size, output_size, input_size, mask_type='output')

            network = nn.Sequential(
                MaskedLinear(input_size, hidden_size, input_mask), nn.ReLU(),
                MaskedLinear(hidden_size, hidden_size, hidden_mask), nn.ReLU(),
                MaskedLinear(hidden_size, output_size, output_mask))
        else:
            network = nn.Sequential(
                nn.Linear(input_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, output_size))

        '''
        for module in network.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                module.bias.data.fill_(0)
        '''

        return network

    def forward(self, inputs):
        """
        Forward.

        Inputs: batch of data.
        Return:
            C
            const
        """
        batch_size = inputs.size(0)
        params = self.network(inputs)
        # params[:,0:kmn] is a, params[:,kmn:end] constant
        # params:b*(kmn+n)
        i = self.k * self.m * self.input_size
        # params
        c = params[:, :i].view(batch_size,
                               -1, self.input_size).transpose(1, 2).view(
            batch_size, self.input_size, self.k, self.m, 1)
        # c: b*n*k*m*1
        const = params[:, i:].view(batch_size, self.input_size)
        # const: b*n
        # C: b*n*k*m*1 X b*n*k *1 *m = b * n* k *m * m
        # every square parts (one of k summed up parts) has m*m parameters
        cc = torch.matmul(c, c.transpose(3, 4))
        return cc, const


#
#   SOS Block:
#

class SOSFlow(nn.Module):
    """
    SOS flow.

    From
    """

    @staticmethod
    def power(z, k):
        """
        Power.

        From
        """
        return z ** (torch.arange(k).float().to(z.device))

    def __init__(self, input_size, hidden_size, k, r, n_layers=1):
        """
        Initialize.

        build conditional nets (cn)/ prepare denominators for polymomials.
        Args:
            input_size:
            hidden_size: number of hidden neurons in every hiden  layer of cn
            k: number of squares (of polynomials)
            r: degree of polynomials
            n_layers: number of hidden layers
        """
        super().__init__()
        self.k = k
        self.m = r + 1

        # build conditioner network
        self.conditioner = ConditionerNet(input_size,
                                          hidden_size, k, self.m, n_layers)
        self.register_buffer('filter', self._make_filter())

    def _make_filter(self):
        # e.g k = 2  m = 3

        n = torch.arange(self.m).unsqueeze(1)
        # n = [[0],[1],[2]]^T
        # n: m*1
        e = torch.ones(self.m).unsqueeze(1).long()
        # e = [[1],[1],[1]]^T
        # e: m*1
        filter = (n.mm(e.transpose(0, 1))) + (e.mm(n.transpose(0, 1))) + 1
        # tensor([[1, 2, 3],
        #         [2, 3, 4],
        #         [3, 4, 5]])
        # filter: m*m
        return filter.float()

    def forward(self, inputs, mode='direct'):
        """
        Forward.

        Args:
        inputs: shape[batch_size,input_size]
        mode: direction of flow
        T-inverse(Z) = S
        """
        batch_size, input_size = inputs.size(0), inputs.size(1)
        # batch_size: b
        # input_size: d
        # inputs: b*d

        C, const = self.conditioner(inputs)
        # C: bs x d x k x m x m
        # const: bs x d

        zz = SOSFlow.power(
            inputs.unsqueeze(-1), self.m).view(batch_size,
                                               input_size, 1, self.m, 1)
        # X: bs x d x 1 x m x 1
        ss = self._transform(zz, C / self.filter) * inputs + const
        # S: bs x d x 1 x m x 1
        # S = T-inverse(X), T-inverse is SoS-flow,
        #  i.e., si =T_i-inverse(x_i)= c+ integral_0^{x_i}
        # (sum of squares dependent on x_1,..x_{i-1})
        # logdet(T-inverse)= log(abs((partial T_1/partial x_1)
        #           *(partial T_2/partial x_2),..(partial T_d/partial x_d)))
        # (partial T_i/partial x_i)= sum of squares (where u = x_i)
        logdet = torch.log(
            torch.abs(self._transform(zz, C))).sum(dim=1, keepdim=True)

        # logdet: log(jacobian of T-inverse)
        # logdet: -log(jacobian of T )
        return ss, logdet

    def _transform(self, xx, cc):
        # C: b* d * k * m * m
        # X: b* d * 1* m * 1
        cc_xx = torch.matmul(cc, xx)  # bs x d x k x m x 1
        xx_cc_xx = torch.matmul(xx.transpose(3, 4), cc_xx)
        # bs x d x k x 1 x 1
        summed = xx_cc_xx.squeeze(-1).squeeze(-1).sum(-1)  # bs x d

        return summed


class BatchNormFlow(nn.Module):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """
    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()
        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, mode='direct'):

        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (
                    inputs - self.batch_mean).pow(2).mean(0) + self.eps
                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)
                self.running_mean.add_(self.batch_mean.data *
                                       (1 - self.momentum))
                self.running_var.add_(self.batch_var.data *
                                      (1 - self.momentum))
                mean = self.batch_mean
                var = self.batch_var

            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta

            # if self.training:
            #     print(f"Train:{mean.mean()},{var.mean()}")
            # else:
            #     print(f"Val:{mean.mean()},{var.mean()}")

            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(
                -1, keepdim=True)
        else:
            if True:  # self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var
            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)
            y = x_hat * var.sqrt() + mean
            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(-1, keepdim=True)
    def _jacob(self, X):
        return None


class Reverse(nn.Module):
    """ An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """
    def __init__(self, num_inputs):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)
    def forward(self, inputs, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(inputs.size(0), 1, device=inputs.device)
    def _jacob(self, X):
        return None

class FlowSequential(nn.Sequential):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def forward(self, inputs, mode='direct', logdets=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, mode)
                logdets += logdet

        return inputs, logdets

    def evaluate(self, inputs):
        N = len(self._modules)
        outputs = torch.zeros(N+1, inputs.size(0), inputs.size(1), device=inputs.device)
        outputs[0,:,:] = inputs
        logdets = torch.zeros(N, inputs.size(0), 1, device=inputs.device)
        for i in range(N):
            outputs[i+1,:,:], logdets[i,:,:] = self._modules[str(i)](outputs[i,:,:], mode='direct')
        return outputs, logdets

    def log_probs(self, inputs):
        u, log_jacob = self(inputs)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)
        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.forward(noise, cond_inputs, mode='inverse')[0]
        return samples

    def jacobians(self, xx):
        assert len(xx.size()) == 1
        N = len(self._modules)
        num_inputs = xx.size(-1)
        jacobians = torch.zeros(N, num_inputs, num_inputs)
        n_jacob = 0
        for i in range(N):
            jj_i = self._modules[str(i)]._jacob(xx)
            if jj_i is not None:
                jacobians[n_jacob, :, :] = jj_i
                n_jacob += 1
            del jj_i
        return jacobians[:n_jacob, :, :]
