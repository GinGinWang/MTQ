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


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as io

# thyroid
class KDDCUP (OneClassDataset):
    
    def __init__(self, n_class = 2, select = 0):

        # type: (str) -> None
        """
        Class constructor.
        :param path: The folder in which to download MNIST.
        """
        super(KDDCUP, self).__init__()
        self.select = select
        self.n_class = n_class
        self.normal_class = 0

        self.name = 'kddcup'
        url_base = "./data/UCI"
        # # KDDCup 10% Data
        url_data = f"{url_base}/kddcup.data_10_percent.gz"
        # info data (column names, col types)
        url_info = f"{url_base}/kddcup.names"

        # Import info data
        df_info = pd.read_csv(url_info, sep=":", skiprows=1, index_col=False, names=["colname", "type"])
        colnames = df_info.colname.values
        coltypes = np.where(df_info["type"].str.contains("continuous"), "float", "str")
        colnames = np.append(colnames, ["status"])
        coltypes = np.append(coltypes, ["str"])

        # Import data
        df = pd.read_csv(url_data, names=colnames, index_col=False,
                         dtype=dict(zip(colnames, coltypes)))
        # Dumminize
        X = pd.get_dummies(df.iloc[:,:-1]).values

        # Create Traget Flag
        # Anomaly data when status is normal, Otherwise, Not anomaly.
        y = np.where(df.status == "normal.", 1, 0)

        # 50 for testing
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.50, random_state=123)

        X_train, y_train = X_train[y_train == 0], y_train[y_train == 0]
        self.X_train, self.y_train = X_train, y_train


        # Transform zone
        self.val_transform = transforms.Compose([ToFloatTensor2Dt()])
        self.test_transform = transforms.Compose([ToFloat32(), OCToFloatTensor2Dt()])
        self.transform = None

        # Other utilities
        self.mode = None
        self.length = None

    def val(self, normal_class):
        # type: (int) -> None
        """
         in validation mode.
        :param normal_class: the class to be considered normal.
        """
        self.mode = 'val'
        self.transform = self.val_transform
        val_num = int(0.1*len(self.X_train))
        self.X_val = self.X_train[0:val_num,:]
        self.y_val = self.y_train[0:val_num]
        self.length = len(self.y_val)


    def train(self,normal_class=0):
        # type: (int) -> None
        """
        By JingJing
        in training mode.

        :param normal_class: the class to be considered normal.
        """
         # normal_class=0
        
        # Update mode, indexes, length and transform
        self.mode = 'train'
        self.transform = self.val_transform
        self.length = len(self.y_train)
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
        self.length = len(self.y_test)
        



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
            # HWC
            x = self.X_test[i,:]
            # x = np.expand_dims(x, axis=0)
            # x = x.reshape((11,11))
            y = (self.y_test[i]==0)
            x = np.float32(x)[..., np.newaxis]
            sample = x, int(y)

        elif self.mode == 'val':
            x= self.X_val[i,:]
            # x = np.expand_dims(x, axis=0)
            # x = x.reshape((11,11))
            x = np.float32(x)[..., np.newaxis]
            sample = x, x
        elif self.mode == 'train':
            x= self.X_train[i,:]
            # x = np.expand_dims(x, axis=0)
            # x = x.reshape((11,11)) 
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
        return 1,121,1
    def __repr__(self):
        return ("ONE-CLASS MNIST (normal class = 0)")
