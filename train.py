import argparse
from argparse import Namespace

from datasets.mnist import MNIST
from models import SOSLSA_MNIST
from models import LSAMNIST
from models import SOSAE_MNIST

from datasets.utils import set_random_seed
from result_helpers import OneClassTrainHelper

import torch.optim as optim

def train_mnist():
    # type: () -> None
    """
    Performs One-class classification tests on MNIST
    """

    # dataset split to train and testcd
    dataset = MNIST(path='data/MNIST')
    print ("dataset shape: "dataset.shape)
    
    # Build Model
    estimator_name = "maf" # specify an estimator
    
    model =SOSLSA_MNIST(input_shape=dataset.shape,code_length=32, num_blocks=5,est_name= estimator_name, coder_name ="LSA").cuda()

    # model = LSAMNIST(input_shape=dataset.shape, code_length=32, cpd_channels=100).cuda()


    # Learning Rate lr
    lr = 0.001

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    # trained model save_dir
    dirName = 'checkpoints/mnist/'

    # Initialize training process
    helper = OneClassTrainHelper(dataset, model, optimizer, checkpoints_dir=dirName, train_epoch=100)

    # Start training 
    helper.train_one_class_classification()

def parse_arguments():
            # type: () -> Namespace
    """
    Argument parser.

    :return: the command line arguments.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', type=str,
                        help='The name of the dataset to perform tests on.'
                        'Choose among `mnist`, `cifar10`, `ucsd-ped2`, `shanghaitech`', metavar='')

    return parser.parse_args()

def main():

        # Parse command line arguments
    args = parse_arguments()

    # Lock seeds
    set_random_seed(30101990)

    # Run test
    if args.dataset == 'mnist':
        train_mnist()
    # elif args.dataset == 'cifar10':
    #     test_cifar()
    else:
        raise ValueError('Unknown dataset')


# Entry point
if __name__ == '__main__':
    main()
