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
# from sklearn.model_selection import train_test_split
import scipy.io as io

# thyroid
class KDDCUP (OneClassDataset):
    
    def __init__(self, path):

        # type: (str) -> None
        """
        Class constructor.
        :param path: The folder in which to download MNIST.
        """
        super(KDDCUP, self).__init__()

        self.name = 'kddcup'
        self.normal_class = 1
        # path='data/UCI'
        # # KDDCup 10% Data
        data = pd.read_csv(f"{path}/kddcup.data_10_percent.gz", header=None,names=['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'type'])

        #categorical: protocol_type, service, flag, land, logged_in,is_host_login,is_guest_login
        #one-hot coding on protocol_type, service, flag
        #binary value on land, logged_in, is_guest_login, is_host_login

        data.loc[data["type"] != "normal.", 'type'] = 1 # as Nominal
        data.loc[data["type"] == "normal.", 'type'] = 0 # as Novel

        # one-hot coding for categorical features
        one_hot_protocol = pd.get_dummies(data["protocol_type"])
        one_hot_service = pd.get_dummies(data["service"])
        one_hot_flag = pd.get_dummies(data["flag"])

        one_hot_land = pd.get_dummies(data["land"])
        one_hot_logged_in= pd.get_dummies(data["logged_in"])
        one_hot_is_guest_login = pd.get_dummies(data["is_guest_login"])
        one_hot_is_host_login  = pd.get_dummies(data["is_host_login"])


        data = data.drop("protocol_type",axis=1)
        data = data.drop("service",axis=1)
        data = data.drop("flag",axis=1)
            
        data = pd.concat([one_hot_protocol, one_hot_service,one_hot_flag,data], axis=1)
        data = pd.concat([one_hot_land,one_hot_is_host_login,one_hot_logged_in,one_hot_is_guest_login, data], axis=1)
        
        features = data.iloc[:,:-1].values
        labels = data.iloc[:,-1].values

        

        N, self.dimension = features.shape
        print(self.dimension)
        
        novel_data = features[labels==0]
        novel_labels = labels[labels==0]

        N_novel = novel_data.shape[0]

        nominal_data = features[labels==1]
        nominal_labels = labels[labels==1]

        N_nominal = nominal_data.shape[0]


        print(f"KDDCUP({N_novel+N_nominal})\t N_novel:{N_novel}\tN_nominal:{N_nominal}\t novel-ratio:{N_novel/N_nominal}")

        randIdx = np.arange(N_nominal)
        np.random.shuffle(randIdx)
        N_train_valid = N_nominal//2
        N_train = int(N_train_valid *0.9)
        N_valid = N_train_valid -N_train

        # 0.45 nominal data
        self.X_train = nominal_data[randIdx[:N_train]]
        self.y_train = nominal_labels[randIdx[:N_train]]
        
        # 0.05 nominal data
        self.X_val = nominal_data[randIdx[N_train:N_train_valid]]
        self.y_val =nominal_data[randIdx[N_train:N_train_valid]]
        
        # 0.5 nominal data + all normal data
        self.X_test = nominal_data[randIdx[N_train_valid:]]
        self.y_test = nominal_labels[randIdx[N_train_valid:]]
        self.X_test = np.concatenate((self.X_test, novel_data),axis=0)
        self.y_test = np.concatenate((self.y_test, novel_labels),axis=0)


        # Other utilities
        self.mode = None
        self.length = None

        # Transform zone
        self.val_transform = transforms.Compose([ToFloatTensor2Dt()])
        self.train_transform = transforms.Compose([ToFloatTensor2Dt()])
        self.test_transform = transforms.Compose([ToFloat32(), OCToFloatTensor2Dt()])
        self.transform = None


        print(f"thyroid: train-f{N_train},validation-{N_valid},test-{len(self.y_test)}(novel-ratio:){float(sum(self.y_test == 1)/len(self.y_test))}")


    def val(self, cl = 1):
        # type: (int) -> None
        """
         in validation mode.
        :param normal_class: the class to be considered normal.
        """
        self.mode = 'val'
        self.length = len(self.y_val)
        self.transform = self.val_transform


    def train(self, cl = 1):
        # type: (int) -> None
        """
        By JingJing
        in training mode.

        :param normal_class: the class to be considered normal.
        """
        # Update mode, indexes, length
        self.mode = 'train'

        self.length = len(self.y_train)
        # manually shuffled
        randIdx = np.arange(self.length)
        self.X_train = self.X_train[randIdx]

        self.transform = self.train_transform


    def test(self, cl = 1):
        # type: (int) -> None
        """
        Sets  test mode.

        :param normal_class: the class to be considered normal.
        :param norvel_ratio: the ratio of novel examples
        """
        # Update mode, length and transform
        self.mode = 'test'

        # testing examples (norm)
        self.length = len(self.y_test)
        self.transform = self.test_transform
        

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
        Returns all test possible test sets ().
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
        # type: () -> Tuple[int, int, int]
        """
        Returns the shape of examples.
        """
        return  self.dimension
    def __repr__(self):
        return ("ONE-CLASS kddcup")
