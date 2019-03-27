import argparse
from argparse import Namespace

from datasets.mnist import MNIST
from models import SOSLSA_MNIST
from models import LSAMNIST
from models import SOSAE_MNIST

from datasets.utils import set_random_seed
from result_helpers import OneClassTrainHelper

import torch.optim as optim



def main():
    # type: () -> None
    """
    Performs One-class classification tests on one dataset
    """

    ## Parse command line arguments
    args = parse_arguments()

    # Lock seeds
    set_random_seed(30101990)

    # prepare dataset in train mode
    if args.dataset == 'mnist':
        dataset = MNIST(path='data/MNIST')
    elif args.dataset == 'cifar10':
        dataset = MNIST(path='data/CIFAR')
    else:
        raise ValueError('Unknown dataset')
    print ("dataset shape: ",dataset.shape)
    

    # Build Model
    if coder_name == "LSA":
        if estimator_name == "SOS"    
        model =LSA_MNIST(input_shape=dataset.shape,code_length=32, num_blocks=5,est_name= arg.estimator_name, coder_name ="LSA").cuda()
    # (add other models here)    

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=arg.lr, weight_decay=1e-6)

    # trained model save_dir
    dirName = f'checkpoints/{arg.dataset}/'
    
    # Initialize training process
    helper = OneClassTrainHelper(dataset, model, optimizer, checkpoints_dir=dirName, train_epoch=arg.epochs)

    # Start training 
    helper.train_one_class_classification()

















def parse_arguments():
    # type: () -> Namespace
    """
    Argument parser.

    :return: the command line arguments.
    """
    parser = argparse.ArgumentParser(description = 'Train autoencoder with density estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # dataset 
    parser.add_argument('dataset', type=str,
                        help='The name of the dataset to perform tests on.'
                        'Choose among `mnist`, `cifar10`', metavar='')
    
    # batch size for training
    parser.add_argument(
    '--batch-size',
    type=int,
    default=100,
    help='input batch size for training (default: 100)')
    
    # epochs 
    parser.add_argument(
    '--epochs',
    type=int,
    default=1000,
    help='number of epochs to train (default: 1000)')
    
    # learning rate 
    parser.add_argument(
    '--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')

    # density estimator
    parser.add_argument(
    '--density', default='MAF', help='flow to use: MAF | SOS')

    # disable cuda
    parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')

    # number of blocks
    parser.add_argument(
    '--num-blocks',
    type=int,
    default=5,
    help='number of invertible blocks (default: 5)')


    # number of blocks
    parser.add_argument(
    '--code-length',
    type=int,
    default=32,
    help='length of hidden vector (default: 32)')

    #K  (only for SOS flow) 
    #M (only for SOS flow)

    return parser.parse_args()
    
# Entry point
if __name__ == '__main__':
    main()
