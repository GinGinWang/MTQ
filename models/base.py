from functools import reduce
from operator import mul

import torch
import torch.nn as nn

class BaseModule(nn.Module):
    """
    Implements the basic module.
    All other modules inherit from this one
    """
    def load_w(self, checkpoint_path):
        # type: (str) -> None
        """
        Loads a checkpoint into the state_dict.

        :param checkpoint_path: the checkpoint file to be loaded.
        """
        self.load_state_dict(torch.load(checkpoint_path))

    def load_lsa(self, checkpoint_path):
        pretrained_dict = torch.load(checkpoint_path)
        model_dict = self.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.load_state_dict(model_dict)


    
    def load_checkpont(self,filename):
        start_epoch = 0

        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            losslogger = checkpoint['losslogger']
            print("=> loaded checkpoint '{}' (epoch {})"
                      .format(filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

        return model, optimizer, start_epoch, losslogger

    def __repr__(self):
        # type: () -> str
        """
        String representation: return a printable representation
        """
        good_old = super(BaseModule, self).__repr__()
        addition = 'Total number of parameters: {:,}'.format(self.n_parameters)

        return good_old + '\n' + addition

    def __call__(self, *args, **kwargs):
        # 
        return super(BaseModule, self).__call__(*args, **kwargs)

    @property
    def n_parameters(self):
        # type: () -> int
        """
        Number of parameters of the model.
        as a property of this class
        """
        n_parameters = 0
        for p in self.parameters():
            if hasattr(p, 'mask'):
                n_parameters += torch.sum(p.mask).item()
            else:
                n_parameters += reduce(mul, p.shape)
        return int(n_parameters)
    