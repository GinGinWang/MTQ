import argparse
from argparse import Namespace

from datasets.mnist import MNIST
from datasets.cifar10 import CIFAR10
from models import LSA_MNIST
from models import LSA_CIFAR10

from datasets.utils import set_random_seed
from train_one_class import OneClassTrainHelper


import torch.optim as optim
import os



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
        lam =0.1

    elif args.dataset == 'cifar10':
        dataset = CIFAR10(path='data/CIFAR10')
        lam = 0.1
    
    else:
        raise ValueError('Unknown dataset')
    
    print ("dataset shape: ",dataset.shape)
    

    




    # trained model save_dir
    dirName = f'checkpoints/{args.dataset}/combined{args.combine_density}/'
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print(f'Make Dir:{dirName}')



    for cl_idx, cl in enumerate(dataset.train_classes):

        dataset.train(cl)
        # Build Model
        if args.autoencoder == "LSA":
            
            if args.dataset == 'mnist':        
                model =LSA_MNIST(input_shape=dataset.shape, code_length=args.code_length, num_blocks=args.num_blocks, est_name= args.estimator, combine_density = args.combine_density).cuda()
            
            elif args.dataset == 'cifar10':
                model =LSA_CIFAR10(input_shape=dataset.shape, code_length=args.code_length, num_blocks=args.num_blocks, est_name= args.estimator, combine_density = args.combine_density).cuda()


        # (add other models here)    

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

        

        
        # Initialize training process
        helper = OneClassTrainHelper(dataset, model, optimizer, lam = lam, checkpoints_dir=dirName, train_epoch=args.epochs, batch_size= args.batch_size)

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
    default=64,
    help='length of hidden vector (default: 32)')

    # join density (For train or test)
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
