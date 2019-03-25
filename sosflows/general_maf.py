import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output

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
    def __init__(self, in_features, out_features, mask, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', mask)

    def forward(self, inputs):
        return F.linear(inputs, self.weight * self.mask, self.bias)


#nn.MaskedLinear = MaskedLinear


class GeneralMADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509s).
    """
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.param_net = self._make_param_net()

    def _make_param_net(self):
        raise NotImplementedError()

    def forward(self, inputs, mode='direct'):
        raise NotImplementedError()

    def _params(self, inputs):
        return self.param_net(inputs)


class MADE(GeneralMADE):
    def __init__(self, num_inputs, num_hidden, use_tanh=True):
        super().__init__(num_inputs, num_hidden, 2*num_inputs)
        self.use_tanh = use_tanh

    def _make_param_net(self):
        '''

        '''
        input_mask = get_mask(
            self.num_inputs, self.num_hidden, self.num_inputs, mask_type='input')
        hidden_mask = get_mask(self.num_hidden, self.num_hidden, self.num_inputs)
        output_mask = get_mask(
            self.num_hidden, self.num_inputs * 2, self.num_inputs, mask_type='output')

        return nn.Sequential(
            nn.MaskedLinear(self.num_inputs, self.num_hidden, input_mask), nn.ReLU(),
            nn.MaskedLinear(self.num_hidden, self.num_hidden, hidden_mask), nn.ReLU(),
            nn.MaskedLinear(self.num_hidden, self.num_inputs * 2, output_mask))

    def forward(self, inputs, mode='inverse'):
        if mode == 'direct':
            return self._direct(inputs)
        else:
            return self._inverse(inputs)

    def _direct(self, inputs):
        params = self.param_net(inputs)
        m, a = params.chunk(2, 1)
        if self.use_tanh:
            a = torch.tanh(a)
        u = (inputs - m) * torch.exp(-a)
        return u, -a.sum(-1, keepdim=True)

    def _inverse(self, inputs):
        x = torch.zeros_like(inputs)
        J = torch.zeros(inputs.size(0), 1)
        for i in range(inputs.shape[1]):
            params = self.param_net(x)
            m, a = params.chunk(2, 1)
            if self.use_tanh:
                a = torch.tanh(a)
            x_i = inputs[:, i] * torch.exp(a[:, i]) + m[:, i]
            J_i = -a[:, i]
            x[:, i] = x_i
            J += J_i.unsqueeze(1)
        return x, J


class SumSqMAF(GeneralMADE):
    def __init__(self, num_inputs, num_hidden, k, m):

        super().__init__(num_inputs, num_hidden, k * m * num_inputs + num_inputs)
        self.k = k
        self.m = m
        # inverse net :
        self.inverse_net = self._make_param_net()
        # register buffer :
        self.register_buffer('filter', self._make_filter())

    @staticmethod
    def power(z, k):
        return z ** (torch.arange(k).float().to(z.device))

    def _make_filter(self):
        n = torch.arange(self.m).unsqueeze(1)
        e = torch.ones(self.m).unsqueeze(1).long()
        filter = (n.mm(e.transpose(0, 1))) + (e.mm(n.transpose(0, 1))) + 1
        return filter.float()

    def _make_param_net(self):
        '''
        Build parameters network
        '''
        # if input dimension >1
        if self.num_inputs > 1:
            input_mask = get_mask(
                self.num_inputs, self.num_hidden, self.num_inputs, mask_type='input')
            hidden_mask = get_mask(self.num_hidden, self.num_hidden, self.num_inputs)
            output_mask = get_mask(
                self.num_hidden, self.num_outputs, self.num_inputs, mask_type='output')

            return nn.Sequential(
                MaskedLinear(self.num_inputs, self.num_hidden, input_mask), nn.ReLU(),
                MaskedLinear(self.num_hidden, self.num_hidden, hidden_mask), nn.ReLU(),
                MaskedLinear(self.num_hidden, self.num_outputs, output_mask))
        
        # if input dimension =1
        else:
            return nn.Sequential(
                nn.Linear(self.num_inputs, self.num_hidden), nn.ReLU(),
                nn.Linear(self.num_hidden, self.num_hidden), nn.ReLU(),
                nn.Linear(self.num_hidden, self.num_outputs))

    def _params(self, inputs, mode='direct'):
        '''
        
        type: tensor[]->tensor[],tensor[]

        '''
        
        batch_size = inputs.size(0)
        
        if mode == 'direct':
            params = self.param_net(inputs)
        else:
            params = self.inverse_net(inputs)
        
        #i1 = self.k * self.num_inputs
        i2 = self.k * self.m * self.num_inputs
        #a = params[:, :i1].view(batch_size, self.num_inputs, self.k)
        #c = params[:, i1:i2].view(batch_size,self.num_inputs, self.k, self.m, 1)   # bs x d x k x m x m
        c = params[:, :i2].view(batch_size, self.num_inputs, self.k, self.m, 1)  # bs x d x k x m x m
        C = torch.matmul(c, c.transpose(3, 4)) / self.filter                        # bs x d x k x m x m
        constant = params[:,i2:].view(batch_size, self.num_inputs)
        #return a, C, constant
        return C, constant

    def forward(self, inputs, mode='direct', train=False):
        '''
        Forward Flow
        '''
        batch_size = inputs.size(0)
        

        #a, C, constant = self._params(inputs, mode=mode)
        C, constant = self._params(inputs, mode=mode)
        #
        X = SumSqMAF.power(inputs.unsqueeze(-1), self.m).view(batch_size, self.num_inputs, 1, self.m, 1)  # bs x d x 1 x m x 1
        XCX = self._transform(X, C)
        Z = XCX * inputs + constant  # bs x d
        if train:
            logdet = torch.zeros(batch_size, 1).to(inputs.device)
        else:
            Cbar = C * self.filter
            #Jacobian |T'(X)|
            logdet = torch.sum(torch.log(torch.abs(self._transform(X, Cbar))), 1, keepdim=True)
        return Z, logdet

    def _transform(self, X, C):
        #batch_size = X.size(0)
        CX = torch.matmul(C, X)                                                                 # bs x d x k x m x 1
        XCX = torch.matmul(X.transpose(3, 4), CX)                                               # bs x d x k x 1 x 1
        #aXCX = torch.matmul(a.unsqueeze(-2), XCX.squeeze(-1)).view(batch_size, self.num_inputs) # bs x d
        return torch.sum(XCX.view(XCX.size()[:3]), 2)


