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

# thyroid
class THYROID (OneClassDataset):
    
    def __init__(self, path, n_class = 2, select = 0):

        # type: (str) -> None
        """
        Class constructor.
        :param path: The folder in which to download MNIST.
        """
        super(THYROID, self).__init__()

        self.path = path
        data = io.loadmat(path)
        self.normal_class = 0
        self.n_class = n_class
        self.select = select
        self.name = 'thyroid'
        self.train_split ={}
        self.test_split = {}
        idxs = np.arange(len(data['y']))
        np.random.shuffle(idxs)

        self.train_split['X'] = data['X'][idxs[0:int(0.5*len(idxs))]]
        self.train_split['y'] = data['y'][idxs[0:int(0.5*len(idxs))]]
        print(f"{len(self.train_split['y'])} for train sets")
        
        self.test_split['X'] = data['X'][idxs[int(0.5*len(idxs)):]]
        self.test_split['y'] = data['y'][idxs[int(0.5*len(idxs)):]]
        print(f"{len(self.test_split['y'])} for  test sets")
        
        self.train_idx = np.arange(len(self.train_split['y']))
        self.test_idx = np.arange(len(self.test_split['y']))


        # Transform zone
        self.val_transform = transforms.Compose([ToFloatTensor2Dt()])
        self.test_transform = transforms.Compose([ToFloat32(), OCToFloatTensor2Dt()])
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
         in validation mode.
        :param normal_class: the class to be considered normal.
        """
        # Update mode, indexes, length and transform 
        self.mode = 'val'
        self.transform = self.val_transform
        
        self.val_idxs = [idx for idx in self.train_idx if self.train_split['y'][idx] == 0 ]
        self.val_idxs =self.val_idxs[int(0.9*len(self.val_idxs)):]
       
        self.length = len(self.val_idxs)



    def train(self, normal_class):
        # type: (int) -> None
        """
        By JingJing
        in training mode.

        :param normal_class: the class to be considered normal.
        """

        # Update mode, indexes, length and transform
        self.mode = 'train'
        self.transform = self.val_transform
        # training examples are all normal
        self.train_idxs = [idx for idx in self.train_idx if self.train_split['y'][idx] == 0]

        
        self.length = len(self.train_idxs)
        # print 
        print(f"Training Set prepared, Num:{self.length}")
#---


    def test(self, normal_class=0, norvel_ratio =1):
        # type: (int) -> None
        """
        Sets  test mode.

        :param normal_class: the class to be considered normal.
        :param norvel_ratio: the ratio of novel examples
        """
        
        # Update mode, length and transform
        self.mode = 'test'
        self.transform = self.test_transform

        # testing examples (norm)
        self.length = len(self.test_split['y'])
        
        normal_idxs = [idx for idx in self.test_idx if self.test_split['y'][idx] == 0]
        normal_num = len(normal_idxs)
        novel_num = self.length - normal_num

        self.test_idxs = self.test_idx
        
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

        # Load the i-th example
        if self.mode == 'test':
            x = self.test_split['X'][self.test_idxs[i]]
            y = self.test_split['y'][self.test_idxs[i]]

            x = np.float32(x)[..., np.newaxis]
            sample = x, int(y==0)

        elif self.mode == 'val':
            x= self.train_split['X'][self.val_idxs[i]]
            x = np.float32(x)[..., np.newaxis]
            sample = x, x
        elif self.mode == 'train':
            x= self.train_split['X'][self.train_idxs[i]]
            x = np.float32(x)[..., np.newaxis]
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
        return 1, 6, 1
    def __repr__(self):
        return ("ONE-CLASS MNIST (normal class =  0 )")
