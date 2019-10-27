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
from datasets.transforms import ToFloatTensor2Dt
from datasets.transforms import OCToFloatTensor2Dt
import scipy.io as io
import os
# thyroid
class THYROID (OneClassDataset):
    
    def __init__(self, path):

        # type: (str) -> None
        """
        Class constructor.
        :param path: The folder in which to download MNIST.
        """
        super(THYROID, self).__init__()

        self.path = path
        data = io.loadmat(f'{path}/thyroid.mat')
        self.normal_class = 1 
        
        self.name = 'thyroid'
        self.train_split ={}
        self.test_split = {}

        features = data['X']
        labels = data['y'].squeeze()
        labels = (labels == 0)

        N, self.dimension = features.shape
        
        nominal_data = features[labels==1,:]
        nominal_labels = labels[labels==1]

        N_nominal = nominal_data.shape[0]

        novel_data = features[labels==0,:]
        novel_labels = labels[labels==0]

        N_novel = novel_data.shape[0]

        print(f"thyroid:{N_novel+N_nominal}\t N_novel:{N_novel}\tN_nominal:{N_nominal}\t novel-ratio:{N_novel/N_nominal}")


        randIdx = np.arange(N_nominal)
        np.random.shuffle(randIdx)
        
        N_train_valid = N_nominal//2
        N_train = int(N_train_valid *0.9)
        N_valid = N_train_valid -N_train

        # 0.45 nominal data as training set
        self.X_train = nominal_data[randIdx[:N_train]]
        self.y_train = nominal_labels[randIdx[:N_train]]
        
        # 0.05 nominal data as validation set
        self.X_val = nominal_data[randIdx[N_train:N_train_valid]]
        self.y_val =nominal_data[randIdx[N_train:N_train_valid]]

        # 0.5 nominal data + all novel data as test set
        self.X_test = nominal_data[randIdx[N_train_valid:]]
        self.y_test = nominal_labels[randIdx[N_train_valid:]]
        self.X_test = np.concatenate((self.X_test, novel_data),axis=0)
        self.y_test = np.concatenate((self.y_test, novel_labels),axis=0)

        
        print(f"thyroid: train-{N_train},validation-{N_valid},test-{len(self.y_test)}(novel-ratio:{float(sum(self.y_test == 0)/len(self.y_test))})")

        # Other utilities
        self.mode = None
        self.length = None

        # Transform zone
        self.val_transform = transforms.Compose([ToFloatTensor2Dt()])
        self.train_transform = transforms.Compose([ToFloatTensor2Dt()])
        self.test_transform = transforms.Compose([ToFloat32(), OCToFloatTensor2Dt()])
        self.transform = None

    def val(self, nominal_class = 1 ):
        # type: (int) -> None
        """
         in validation mode.
        :param nominal_class: the class to be considered nominal.
        """
        # Update mode, indexes, length and transform 
        self.mode = 'val'
        self.length = len(self.y_val)
        self.transform = self.val_transform

    def train(self, nominal_class = 1):
        # type: (int) -> None
        
        self.mode = 'train'
        self.length = len(self.y_train)
         # manually shuffled
        randIdx = np.arange(self.length)
        self.X_train = self.X_train[randIdx]

        self.transform = self.train_transform




    def test(self, nominal_class=1):
        # type: (int) -> None
        """
        Sets  test mode.

        :param nominal_class: the class to be considered nominal.
        :param norvel_ratio: the ratio of novel examples
        """
        
        # Update mode, length and transform
        self.mode = 'test'

        # testing examples (norm)
        self.length = len(self.y_test)
        self.transform = self.test_transform


    def __len__(self):
        # type: () -> int
        """The size of mini-batches is 1024. The learning rate is $10^{-5}$. The training process is stopped after 100 epochs of non-decreasing loss.
        Returns the number of examples.
        """
        return self.length

    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]
        """
        Provides the i-th example.
        """

        if self.mode == 'train':
            x = self.X_train[i]
            x = np.float32(x)[..., np.newaxis]
            sample = x, x

        elif self.mode == 'val':
            x = self.X_val[i]
            x = np.float32(x)[..., np.newaxis]
            sample = x, x

        elif self.mode == 'test':
            x = self.X_test[i]
            y = self.y_test[i]
            x = np.float32(x)[..., np.newaxis]
            sample = x, y

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
        return [1]

    @property
    def train_classes(self):
        # type: () -> np.ndarray
        """
        Returns all test possible test sets (the 10 classes).
        """
        return [1]

    @property
    def shape(self):
        # type: () -> Tuple[int, int]
        """
        Returns the shape of examples.
        """
        return 1,6,1
    def __repr__(self):
        return ("ONE-CLASS MNIST (nominal class =  0 )")
