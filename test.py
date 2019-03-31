import argparse
from argparse import Namespace

from datasets.mnist import MNIST
from models import LSA_MNIST

from datasets.utils import set_random_seed
from result_helpers import OneClassTestHelper

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
        model =LSA_MNIST(input_shape=dataset.shape,code_length=32, num_blocks=5,est_name= args.estimator,combine_density= args.combine_density).cuda().eval()
    
    # (add other models here)    
   
    # trained model save_dir
    dirName = f'checkpoints/{args.dataset}/combined{args.combine_density}/'

    # Initialize training process
    helper = OneClassTestHelper(dataset, model, args.score_normed, args.novel_ratio, checkpoints_dir= dirName, output_file= f"{args.autoencoder}_{args.estimator}_{args.dataset}_cd{args.combine_density}_nml{args.score_normed}_nlration{args.novel_ratio}")

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
    '--batch_size',
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
    '--no_cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')

    # number of blocks
    parser.add_argument(
    '--num_blocks',
    type=int,
    default=5,
    help='number of invertible blocks (default: 5)')


    # number of blocks
    parser.add_argument(
    '--code_length',
    type=int,
    default=32,
    help='length of hidden vector (default: 32)')

    # Normalize the novelty score
    parser.add_argument(
        '--score_normed',
        default= True,
        help ='For Test: Normalize novelty score by Valid Set' )

    # novel ratio
    # default use 10% novel examples in test set
    parser.add_argument(
        '--novel_ratio',
        type = float,
        default= 0.1,
        help ='For Test: Ratio, novel examples in test sets: [0,1,0.5]' )
    
    # join density
    parser.add_argument(
        '--combine_density',
        default= False,
        help = 'Combine reconstruction loss in the input of density estimator'
        )

    #K  (only for SOS flow) 
    
    #M (only for SOS flow)

    return parser.parse_args()

# Entry point
if __name__ == '__main__':
    main()
