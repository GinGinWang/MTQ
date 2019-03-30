from typing import Tuple
from typing import Union

import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms

from datasets.base import OneClassDataset
from datasets.transforms import OCToFloatTensor2D
from datasets.transforms import ToFloat32
from datasets.transforms import ToFloatTensor2D


class MNIST(OneClassDataset):
    """
    Models MNIST dataset for one class classification.
    """
    def __init__(self, path):
        # type: (str) -> None
        """
        Class constructor.

        :param path: The folder in which to download MNIST.
        """
        super(MNIST, self).__init__()

        self.path = path

        self.normal_class = None

        # Get train and test split
        self.train_split = datasets.MNIST(self.path, train=True, download=True, transform=None)

        self.test_split = datasets.MNIST(self.path, train=False, download=True, transform=None)

        # Shuffle training indexes to build a validation set (see val())
        train_idx = np.arange(len(self.train_split))
        np.random.shuffle(train_idx)
        self.shuffled_train_idx = train_idx

        # Shuffle testing indexes to build  a test set (see test())
        test_idx = np.arange(len(self.test_split))
        np.random.shuffle(test_idx)
        self.shuffled_test_idx = test_idx

        # Transform zone
        self.val_transform = transforms.Compose([ToFloatTensor2D()])
        self.test_transform = transforms.Compose([ToFloat32(), OCToFloatTensor2D()])
        self.transform = None

        # Other utilities
        self.mode = None
        self.length = None

        # val idx in normal class (all possible classes)
        self.val_idxs = None
        # train idx in normal class
        self.train_idxs = None
        # test idx with 50%->90% normal class(50% -> 10% novelty)
        self.test_idxs = None 


    def val(self, normal_class):
        # type: (int) -> None
        """
        Sets MNIST in validation mode.

        :param normal_class: the class to be considered normal.
        """
        self.normal_class = int(normal_class)

        # Update mode, indexes, length and transform
        self.mode = 'val'
        self.transform = self.val_transform
        
        self.val_idxs = self.shuffled_train_idx[int(0.9 * len(self.shuffled_train_idx)):]

        # valid examples (90% of training examples)
        self.val_idxs = [idx for idx in self.val_idxs if self.train_split[idx][1] == self.normal_class]
        self.length = len(self.val_idxs)
#--------------------------------------------------------------------
    def train(self, normal_class):
        # type: (int) -> None
        """
        By JingJing
        Sets MNIST in training mode.

        :param normal_class: the class to be considered normal.
        """
        self.normal_class = int(normal_class)

        # Update mode, indexes, length and transform
        self.mode = 'train'
        self.transform = self.val_transform
        # training examples are all normal
        self.train_idxs = [idx for idx in self.shuffled_train_idx if self.train_split[idx][1] == self.normal_class]

        self.length = len(self.train_idxs)
        # print 
        print(f"Training Set prepared, Num:{self.length}")
#---------------------------------------------------------------------

    def test(self, normal_class, novel_ratio):
        # type: (int) -> None
        """
        Sets MNIST in test mode.

        :param normal_class: the class to be considered normal.
        :param norvel_ratio: the ratio of novel examples
        """
        self.normal_class = int(normal_class)

        # Update mode, length and transform
        self.mode = 'test'
        self.transform = self.test_transform

        # create test examples (normal)
        self.test_idxs = [idx for idx in self.shuffled_test_idx if self.gest_split[idx][1] == self.normal_class]
        normal_num = len(self.test_idxs)

        # add test examples (unnormal)
        novel_num  = normal/(1-novel_ratio) - normal_num
        
        novel_idx = [idx for idx in self.shuffled_test_idx[novel_num:] if self.gest_split[idx][1] != self.normal_class]

        self.test_idxs = [self.test_idxs, novel_idx]
        
        # testing examples (norm)
        self.length = len(self.test_idxs)

        print(f"Test Set prepared, Num:{self.length},Novel_num:{len(novel_num)},Normal_num:{normal_num}")


    def __len__(self):
        # type: () -> int
        """
        Returns the number of examples.
        """
        return self.length

    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]
        """
        Provides the i-th example.
        """
        assert self.normal_class is not None, 'Call test() first to select a normal class!'

        # Load the i-th example
        if self.mode == 'test':
            x, y = self.test_split[i]
            x = np.uint8(x)[..., np.newaxis]
            sample = x, int(y == self.normal_class)

        elif self.mode == 'val':
            x, _ = self.train_split[self.val_idxs[i]]
            x = np.uint8(x)[..., np.newaxis]
            sample = x, x
        elif self.mode == 'train':
            x, _ = self.train_split[self.train_idxs[i]]
            x = np.uint8(x)[..., np.newaxis]
            sample = x, x
        else:
            raise ValueError

        # Apply transform
        if self.transform:
            sample = self.transform(sample)

        return sample

    @property
    def test_classes(self):
        # type: () -> np.ndarray
        """
        Returns all test possible test sets (the 10 classes).
        """
        return np.arange(0, 10)

    @property
    def train_classes(self):
        # type: () -> np.ndarray
        """
        Returns all test possible test sets (the 10 classes).
        """
        return np.arange(0, 10)


    @property
    def shape(self):
        # type: () -> Tuple[int, int, int]
        """
        Returns the shape of examples.
        """
        return 1, 28, 28

    def __repr__(self):
        return ("ONE-CLASS MNIST (normal class =  {} )").format(self.normal_class)
