from typing import Tuple
from typing import Union

import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms

from datasets.base import OneClassDataset
import os
from datasets.transforms import OCToFloatTensor2D
from datasets.transforms import ToFloat32
from datasets.transforms import ToFloatTensor2D
from datasets.transforms import ToFloatTensor1D

from datasets.transforms import OCToFloatTensor1D


data_dir = '/data/cifar10-gth'

class CIFAR10_GTH(OneClassDataset):
    """
    Models CIFAR10 dataset for one class classification.
    """

    def __init__(self, path, n_class = 10, select= None):

        # type: (str) -> None
        """
        Class constructor.

        :param path: The folder in which to download CIFAR10.
        """
        super(CIFAR10_GTH, self).__init__()

        self.path = path
        self.n_class = n_class

        self.n_class = n_class
        self.select = select
        self.name ='cifar10_gth'
        self.normal_class = None

        self.test_split = None

        self.train_split = None

        # Shuffle testing indexes to build  a test set (see test())
        # Transform zone
        self.val_transform = transforms.Compose([ToFloat32(),ToFloatTensor1D()])
        self.test_transform = transforms.Compose([ToFloat32(), OCToFloatTensor1D()])

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

        # new_data_name = f'cifar10_04_gth_{normal_class}_train_xy.npy'
        new_data_name = f'cifar10_gth_{normal_class}_train_xy.npy'
        self.train_split = np.load(os.path.join(data_dir, new_data_name),allow_pickle = True).item()
        self.val_idxs = np.arange(len(self.train_split))

        self.val_idxs = self.val_idxs[int(0.9 * len(self.val_idxs)):]
        # self.val_idxs = [idx for idx in self.val_idxs if self.train_split[idx][1] == self.normal_class]
        
        self.length = len(self.val_idxs)
        
        print(f'Valset prepared, Num:{self.length}')


    def train(self, normal_class, noise_ratio=0,noise2_ratio=0,noise3_ratio=0,noise4_ratio=0):
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
        # new_data_name = f'cifar10_03_gth_{normal_class}_train_xy.npy'
        new_data_name = f'cifar10_gth_{normal_class}_train_xy.npy'

        print(new_data_name)
        self.train_split = np.load(os.path.join(data_dir, new_data_name),allow_pickle = True).item()

        self.train_idxs = np.arange(len(self.train_split))
        np.sf
        self.train_idxs = self.train_idxs[0:int(0.9*len(self.train_idxs))]
        
        self.length = len(self.train_idxs)
        # print 


        print(f"Training Set prepared, Num:{self.length}")


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
        
        # test_data_name = f'cifar10_m{normal_class}_e03_gth_test.npy'
        test_data_name = f'cifar10_gth_test.npy'

        self.test_split = np.load(os.path.join(data_dir,test_data_name), allow_pickle = True).item()
        test_idx = np.arange(len(self.test_split))
        

        # for i in np.arange(10):
        #         class_num_i =len([idx for idx in test_idx if self.test_split[idx][1]==i])
        #         print(f"class {i}:{class_num_i}")
        
        self.test_idxs = test_idx

        if novel_ratio == 1:
            # testing examples (norm)
            self.length = len(self.test_split)
            normal_idxs = [idx for idx in test_idx if self.test_split[idx][1] == self.normal_class]
            normal_num = len(normal_idxs)
            novel_num = self.length - normal_num

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
            x, y = self.test_split[self.test_idxs[i]]
            sample = x, int(y == self.normal_class)

        elif self.mode == 'val':
            x, _ = self.train_split[self.val_idxs[i]]
            sample = x, x

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
            classes = np.arange(0, self.n_class)
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
        return 1, 72, 256

    def __repr__(self):
        return f"ONE-CLASS CIFAR10 (normal class =  {self.normal_class} )"
