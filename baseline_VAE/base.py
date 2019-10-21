from abc import ABCMeta
from abc import abstractmethod

import numpy as np
from torch.utils.data import Dataset


class DatasetBase(Dataset):
    """
    Base class for all datasets.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def test(self, *args):
        """
        Sets the dataset in test mode.
        """
        pass

    @property
    @abstractmethod
    def shape(self):
        """
        Returns the shape of examples.
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        Returns the number of examples.
        """
        pass

    @abstractmethod
    def __getitem__(self, i):
        """
        Provides the i-th example.
        """
        pass


class OneClassDataset(DatasetBase):
    """
    Base class for all one-class classification datasets.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def val(self, *args):
        """
        Sets the dataset in validation mode.
        """
        pass
    @abstractmethod
    def train(self, *args):
        """
        Sets the dataset in validation mode.
        """
        pass
    
    @property
    @abstractmethod
    def test_classes(self):
        """
        Returns all test possible test classes.
        """
        pass
