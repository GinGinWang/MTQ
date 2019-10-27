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
    def __init__(self, path, n_class = 10, select = None, select_novel_classes = None):

        # type: (str) -> None

        """
        Class constructor.
        :param path: The folder in which to download MNIST.

        """
        super(MNIST, self).__init__()

        self.path = path
        self.n_class = n_class
        self.normal_class = None
        self.select = select


        if select_novel_classes != None:
            self.select_novel_classes = [int(novel_class) for novel_class in select_novel_classes]
        else:
            self.select_novel_classes = None
        
        self.name ='mnist'

        # Get train and test split
        self.train_split = datasets.MNIST(self.path, train=True, download=True, transform=None)
        self.test_split = datasets.MNIST(self.path, train=False, download=True, transform=None)

        # Shuffle training indexes to build a validation set (see val())
        train_idx = np.arange(len(self.train_split))
        np.random.shuffle(train_idx)
        self.shuffled_train_idx = train_idx

        # Transform zone
        # self.val_transform = transforms.Compose([ToFloatTensor2D()])
        self.val_transform = transforms.Compose([ToFloat32(), OCToFloatTensor2D()])
        # self.train_transform = transforms.Compose([ToFloatTensor2D()])
        self.train_transform = transforms.Compose([ToFloat32(), OCToFloatTensor2D()])
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

        self.val_idxs = [idx for idx in self.shuffled_train_idx if self.train_split[idx][1] == self.normal_class]
        self.val_idxs =self.val_idxs[int(0.9*len(self.val_idxs)):]

        self.length = len(self.val_idxs)

    def train(self, normal_class):
        # type: (int) -> None
        """
        Sets MNIST in training mode.
        :param normal_class: the class to be considered normal.
        """
        self.normal_class = int(normal_class)

        # Update mode, indexes, length and transform
        self.mode = 'train'
        self.transform = self.train_transform

        # training examples are all normal 
        self.train_idxs = [idx for idx in self.shuffled_train_idx if self.train_split[idx][1] == self.normal_class]

        self.train_idxs = self.train_idxs[0:int(0.9*len(self.train_idxs))]
       
        # fix the size of training set, noise from other datasets
        self.length = len(self.train_idxs)
        # shuffle the training set manually
        np.random.shuffle(self.train_idxs)

        print(f"Training Set prepared, Num:{self.length}")


    def test(self, normal_class):
        # type: (int) -> None
        """
        Sets MNIST in test mode.

        :param normal_class: the class to be considered normal.
        :param norvel_ratio: the ratio of novel examples
        """
        # select all classes from dataset
        self.normal_class = int(normal_class)

        if self.select_novel_classes == None:
            test_classes = range(10)
        else:
            test_classes = self.select_novel_classes
            test_classes.append(normal_class)
        
        print(f"Test set contains {test_classes}")

        # filter testing indexes to build  a test set (see test())
        test_idx = np.arange(len(self.test_split))
        test_idx = [idx for idx in test_idx if self.test_split[idx][1] in test_classes]

        # Update mode, length and transform
        self.mode = 'test'
        self.transform = self.test_transform

        # testing examples (norm)
        normal_idxs = [idx for idx in test_idx if self.test_split[idx][1] == self.normal_class]
        normal_num = len(normal_idxs)
        self.test_idxs = test_idx
        self.length = len(self.test_idxs)
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
            x = np.uint8(x)[..., np.newaxis]
            sample = x, int(y == self.normal_class)

        elif self.mode == 'val':
            x, y = self.train_split[self.val_idxs[i]]
            x = np.uint8(x)[..., np.newaxis]
            sample = x, int(y == self.normal_class)

        elif self.mode == 'train':
            x, y = self.train_split[self.train_idxs[i]]
            x = np.uint8(x)[..., np.newaxis]
            sample = x, int(y == self.normal_class)
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
            classes = np.arange(0, self.n_class)
        else:
            classes = [self.select]

        return classes


    @property
    def shape(self):
        # type: () -> Tuple[int, int, int]
        """
        Returns the shape of examples.
        """
        return 1, 28, 28

    def __repr__(self):
        return ("ONE-CLASS MNIST (normal class =  {} )").format(self.normal_class)