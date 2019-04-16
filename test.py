import argparse
from argparse import Namespace

from datasets.mnist import MNIST
from datasets.cifar10 import CIFAR10

# LSA
from models import LSA_MNIST
from models import LSA_CIFAR10

# Estimator
from models.estimator_1D import Estimator1D
from models.transform_maf import TinvMAF
from models.transform_sos import TinvSOS

from datasets.utils import set_random_seed


from result_helpers import OneClassTestHelper
import os

import torch

def main():
    # type: () -> None
    """
    Performs One-class classification tests on one dataset
    """

    ## Parse command line arguments
    args = parse_arguments()
    device = torch.device("cuda:0")
    # Lock seeds
    set_random_seed(30101990)

    # prepare dataset in train mode
    if args.dataset == 'mnist':
        dataset = MNIST(path='data/MNIST', n_class = args.n_class, select = args.select)

    elif args.dataset == 'cifar10':
        dataset = CIFAR10(path='data/CIFAR', n_class = args.n_class, select= args.select)
    
    else:
        raise ValueError('Unknown dataset')
    
    
    print ("dataset shape: ",dataset.shape)

    dirName = f'checkpoints/{args.dataset}/combined{args.cd}/'

    c, h , w = dataset.shape

    # Build Model
    # Build Model
    if (not args.coder):
        # directly estimate density by model

        # build Density Estimator
        if args.estimator == 'MAF':
            model = TinvMAF(args.num_blocks, c*h*w,128,args.hidden_size).cuda()

        elif args.estimator == 'SOS':
            model = TinvSOS(args.num_blocks, c*h*w,args.hidden_size).cuda()

    # 1-D estimator from LSA
        elif args.estimator == 'EN':
            self.model = Estimator1D(
            code_length=c*h*w,
            fm_list=[32, 32, 32, 32],
            cpd_channels=100).cuda()
        else:
            raise ValueError('Unknown Estimator')
        
        print (f'No Autoencoder, only use Density Estimator: {args.estimator}')


    else:
        if args.autoencoder == "LSA":
            print(f'Autoencoder:{args.autoencoder}')
            print(f'Density Estimator:{args.estimator}')
            
            if args.dataset == 'mnist': 

                model =LSA_MNIST(input_shape=dataset.shape, code_length=args.code_length, num_blocks=args.num_blocks, est_name = args.estimator, combine_density = args.cd,hidden_size= args.hidden_size)
            
            elif args.dataset == 'cifar10':
            
                model =LSA_CIFAR10(input_shape=dataset.shape, code_length=args.code_length, num_blocks=args.num_blocks, est_name= args.estimator, combine_density = args.cd,hidden_size= args.hidden_size)
        else:
            raise ValueError('Unknown MODEL')
    
    # set to Test mode
    model.to(device).eval()
    
    # Initialize training process
    helper = OneClassTestHelper(dataset, model, args.score_normed, args.novel_ratio, lam = args.lam, checkpoints_dir= dirName, output_file= f"results/{model.name}_{args.dataset}_cd{args.cd}_nml{args.score_normed}_nlration{args.novel_ratio}",device = device)

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
    parser.add_argument('--estimator', type=str, help='The name of density estimator.'
                        'Choose among `SOS`, `MAF`', metavar='')
    # dataset 
    parser.add_argument('--dataset', type=str,
                        help='The name of the dataset to perform tests on.'
                        'Choose among `mnist`, `cifar10`', metavar='')
    
    parser.add_argument('--Combine_density', dest='cd',action = 'store_true',default = False)

    parser.add_argument('--NoAutoencoder', dest='coder',action='store_false', default = True)
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
    default= False,
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
    default=64,
    help='length of hidden vector (default: 32)')

    # number of blocks
    parser.add_argument(
    '--hidden_size',
    type=int,
    default=1024,
    help='length of hidden vector (default: 32)')


    # Normalize the novelty score
    parser.add_argument(
        '--score_normed',
        action ='store_true',
        default= False,
        help ='For Test: Normalize novelty score by Valid Set' )

    # novel ratio
    # default use 10% novel examples in test set
    parser.add_argument(
        '--novel_ratio',
        type = float,
        default= 1,
        help ='For Test: Ratio, novel examples in test sets: [0.1,1]' )
    
   
    parser.add_argument(
    '--n_class',
    type = int,
    default = 10,
    help = 'Number of classes used in experiments')

    parser.add_argument(
    '--select',
    type = int,
    default = None,
    help = 'Select one specific class for training')

    #K  (only for SOS flow) 
    
    #M (only for SOS flow)
    parser.add_argument(
    '--lam',
    type=float,
    default=1,
    help='tradeoff between reconstruction loss and autoregression loss')
    

    return parser.parse_args()

# Entry point
if __name__ == '__main__':
    main()
