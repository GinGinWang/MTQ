import argparse
from argparse import Namespace

from datasets.mnist import MNIST
from models import LSA_MNIST

from datasets.utils import set_random_seed
from result_helpers import OneClassResultHelper

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
    
    

    # Build Model
    if args.autoencoder == "LSA":
        model =LSA_MNIST(input_shape=dataset.shape,code_length=32, num_blocks=5,est_name= args.estimator).cuda().eval()
    # (add other models here)    
   
    # trained model save_dir
    dirName = "checkpoints/mnist/"

    
    # Initialize training process
    helper = OneClassResultHelper(dataset, model,checkpoints_dir= dirName, output_file= f"{args.autoencoder}_{args.estimator}_{args.dataset}" )

    # Start training 
    helper.test_one_class_classification()

















def parse_arguments():
    # type: () -> Namespace
    """
    Argument parser.

    :return: the command line arguments.
    """
    parser = argparse.ArgumentParser(description = 'Train autoencoder with density estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # autoencoder name 
    parser.add_argument('--autoencoder', type=str,
                        help='The Autoencoder framework.'
                        'Choose among `LSA`', metavar='')
    # density estimator
    parser.add_argument('--estimator', type=str, default='SOS', help='The name of density estimator.'
                        'Choose among `SOS`, `MAF`', metavar='')
    # dataset 
    parser.add_argument('--dataset', type=str,
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
