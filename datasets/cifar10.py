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


class CIFAR10(OneClassDataset):
    """
    Models CIFAR10 dataset for one class classification.
    """

    def __init__(self, path, n_class = 10, select= None):

        # type: (str) -> None
        """
        Class constructor.

        :param path: The folder in which to download CIFAR10.
        """
        super(CIFAR10, self).__init__()

        self.path = path
        self.n_class = n_class

        self.n_class = n_class
        self.select = select
        self.name ='cifar10'
        self.normal_class = None


        # Get train and test split
        self.train_split = datasets.CIFAR10(self.path, train=True, download=True, transform=None)

        self.test_split = datasets.CIFAR10(self.path, train=False, download=True, transform=None)

        # Shuffle training indexes to build a training set (see train())
        self.train_idx = np.arange(len(self.train_split))
        np.random.shuffle(self.train_idx)
        self.shuffled_train_idx = self.train_idx

        # Shuffle testing indexes to build  a test set (see test())
        self.test_idx = np.arange(len(self.test_split))
        # np.random.shuffle(test_idx)
        # self.shuffled_test_idx = test_idx

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
        Sets CIFAR10 in validation mode.

        :param normal_class: the class to be considered normal.
        """
        self.normal_class = int(normal_class)

        # Update mode, indexes, length and transform
        self.mode = 'val'
        self.transform = self.val_transform

        # self.val_idxs = [idx for idx in self.train_idx if self.train_split[idx][1] == self.normal_class]
        # self.val_idxs =self.val_idxs[int(0.9*len(self.val_idxs)):]

        self.val_idxs = self.shuffled_train_idx[int(0.9 * len(self.shuffled_train_idx)):]
        self.val_idxs = [idx for idx in self.val_idxs if self.train_split[idx][1] == self.normal_class]
        
        self.length = len(self.val_idxs)
        
        print(f'Valset prepared, Num:{self.length}')


    def val2(self, normal_class):
        # type: (int) -> None
        """
        Sets CIFAR10 in validation mode.

        :param normal_class: the class to be considered normal.
        """
        # Update mode, indexes, length and transform
        self.normal_class = int(normal_class)
        self.mode = 'val2'
        self.transform = self.test_transform
        self.val_idxs = self.shuffled_train_idx[int(0.9 * len(self.shuffled_train_idx)):]
        
        self.length = len(self.val_idxs)

        print(f'Val2 Set prepared, Num:{self.length}')
#--------------------------------------------------------------------
    def train(self, normal_class):
        # type: (int) -> None
        """
        By JingJing
        Sets CIFAR10 in training mode.

        :param normal_class: the class to be considered normal.
        """
        self.normal_class = int(normal_class)

        # Update mode, indexes, length and transform
        self.mode = 'train'
        self.transform = self.val_transform
        
        # training examples are all normal
        # self.train_idxs = [idx for idx in self.train_idx if self.train_split[idx][1] == self.normal_class]
        # self.train_idxs = self.train_idxs[0:int(0.9*len(self.train_idxs))]


        self.train_idxs = self.shuffled_train_idx[0:int(0.9 * len(self.shuffled_train_idx))]
        self.train_idxs = [idx for idx in self.train_idxs if self.train_split[idx][1] == self.normal_class]
        
        
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

        if novel_ratio == 1:
            # testing examples (norm)
            self.length = len(self.test_split)
            normal_idxs = [idx for idx in self.test_idx if self.test_split[idx][1] == self.normal_class]
            normal_num = len(normal_idxs)
            novel_num = self.length - normal_num
        else:
            # create test examples (normal)
            self.test_idxs = [idx for idx in self.test_idx if self.test_split[idx][1] == self.normal_class]

            normal_num = len(self.test_idxs)

            # add test examples (unnormal)
            novel_num  = int(normal_num/(1-novel_ratio) - normal_num)
            
            novel_idxs = [idx for idx in self.test_idx if self.test_split[idx][1] != self.normal_class]

            novel_idxs = novel_idxs[0:novel_num]

            # combine normal and novel part
            self.test_idxs = self.test_idxs+novel_idxs
            
            # testing examples (norm)
            self.length = len(self.test_idxs)

        print(f"Test Set prepared, Num:{self.length},Novel_num:{novel_num},Normal_num:{normal_num}")


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
            sample = x, int(y == self.normal_class)

        elif self.mode == 'val':
            x, _ = self.train_split[self.val_idxs[i]]
            sample = x, x
        elif self.mode == 'val2':
            x, y = self.train_split[i]
            sample = x, int(y == self.normal_class)

        elif self.mode == 'train':
            x, _ = self.train_split[self.train_idxs[i]]
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
        if self.select ==None:
            classes = np.arange(0,self.n_class)
        else:
            classes = [self.select] # select one class to train
        return classes

    @property
    def train_classes(self):
        # type: () -> np.ndarray
        """
        Returns all test possible test sets (the 10 classes).
        """
        if self.select == None:
            classes = np.arange(0,self.n_class)
        else:
            classes = [self.select]
        return classes

    @property
    def shape(self):
        # type: () -> Tuple[int, int, int]
        """
        Returns the shape of examples.
        """
        return 3, 32, 32

    def __repr__(self):
        return f"ONE-CLASS CIFAR10 (normal class =  {self.normal_class} )"
