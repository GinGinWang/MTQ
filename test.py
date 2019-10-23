
import argparse
from argparse import Namespace
import os
import torch
import numpy as np
import torch.nn as nn

from datasets import *

# models
from models.LSA_mnist import LSA_MNIST #  mnist/fmnist
from models.LSA_mnist_deep import LSA_MNIST_D
from models.LSA_mnist_wide import LSA_MNIST_W


from models import LSA_CIFAR10 # cifar10
from models import LSA_KDDCUP # kddcup
from models import LSA_THYROID

from models.estimator_1D import Estimator1D
from models.transform_maf import TinvMAF
from models.transform_sos import TinvSOS

# random seed
from utils import set_random_seed
# Main Class
from result_helpers.test_one_class import OneClassTestHelper

# dir/path 
from utils import create_checkpoints_dir
from utils import create_file_path

def main():
    # type: () -> None
    """
    Performs One-class classification tests on one dataset
    """

    ## Parse command line arguments
    args = parse_arguments()
    device = torch.device('cuda')

    #random seed
    set_random_seed(args.seed) # good mnist

    # prepare dataset in train mode
    if args.dataset == 'mnist':
        dataset = MNIST(path='data/MNIST', n_class = args.n_class, select = args.select, select_novel_classes = args.select_novel_classes)

    elif args.dataset == 'fmnist':
        dataset = FMNIST(path='data/FMNIST', n_class = args.n_class, select = args.select)

    elif args.dataset == 'cifar10':
        dataset = CIFAR10(path='data/CIFAR', n_class = args.n_class, select= args.select)

    elif args.dataset == 'thyroid':
        dataset = THYROID(path ='data/UCI')

    elif args.dataset == 'kddcup':
        dataset = KDDCUP(path = 'data/UCI')
    else:
        raise ValueError('Unknown dataset')


    checkpoints_dir = create_checkpoints_dir(args.dataset, args.fixed, args.mulobj, args.num_blocks, args.hidden_size, args.code_length, args.estimator)

    # MODEL
    if (args.autoencoder == None):
        # directly estimate density by estimator
        print (f'No Autoencoder, only use Density Estimator: {args.estimator}')
        c,h,w = dataset.shape

        # build Density Estimator
        if args.estimator == 'SOS':
            model = TinvSOS(args.num_blocks, c*h*w,args.hidden_size).cuda()

        # 1-D estimator from LSA
        elif args.estimator == 'EN':
            self.model = Estimator1D(
            code_length=c*h*w,
            fm_list=[32, 32, 32, 32],
            cpd_channels=100).cuda()
        
        else:
            raise ValueError('Unknown Estimator')
        
    else:
        if args.autoencoder == "LSA":
            print(f'Autoencoder:{args.autoencoder}')
            print(f'Density Estimator:{args.estimator}')
            
            if args.dataset in ['mnist','fmnist']: 

                model =LSA_MNIST(input_shape=dataset.shape, code_length=args.code_length, num_blocks=args.num_blocks, est_name = args.estimator,hidden_size= args.hidden_size).cuda()

            elif args.dataset in ['kddcup']:
                model = LSA_KDDCUP(num_blocks=args.num_blocks, hidden_size= args.hidden_size, code_length = args.code_length,est_name =args.estimator).cuda()

            elif args.dataset in ['thyroid']:
                model =LSA_THYROID(num_blocks=args.num_blocks, hidden_size= args.hidden_size, code_length = args.code_length,est_name =args.estimator).cuda()
            
            elif args.dataset =='cifar10':
                model =LSA_CIFAR10(input_shape=dataset.shape, code_length = args.code_length, num_blocks=args.num_blocks, est_name= args.estimator,hidden_size= args.hidden_size).cuda()
            else:
                ValueError("Unknown Dataset")        
        
        elif args.autoencoder =='LSAD':
                model =LSA_MNIST_D(input_shape=dataset.shape, code_length=args.code_length, num_blocks=args.num_blocks, est_name = args.estimator,hidden_size= args.hidden_size).cuda()
        
        elif args.autoencoder =='LSAW':
                model =LSA_MNIST_W(input_shape=dataset.shape, code_length=args.code_length, num_blocks=args.num_blocks, est_name = args.estimator,hidden_size= args.hidden_size).cuda()

        
        else:
            raise ValueError('Unknown Autoencoder')


    # # set to Test mode
    result_file_path = create_file_path(args.mulobj, args.fixed, args.pretrained, model.name,args.dataset,args.score_normed,args.num_blocks,args.hidden_size,args.code_length, args.lam, args.checkpoint)

    

    print(checkpoints_dir)

    helper = OneClassTestHelper(
        dataset = dataset, 
        model = model, 
        score_normed = args.score_normed, 
        lam = args.lam, 
        checkpoints_dir = checkpoints_dir, 
        result_file_path = result_file_path, 
        batch_size = args.batch_size, 
        lr = args.lr, 
        epochs = args.epochs, 
        before_log_epochs = args.before_log_epochs, 
        code_length = args.code_length,
        mulobj = args.mulobj, 
        test_checkpoint = args.checkpoint,
        log_step = args.log_step,
        device = device,
        fixed = args.fixed,
        pretrained = args.pretrained,
        load_lsa =args.load_lsa
        )
    
    if args.trainflag:
        cl = args.select
        helper.train_one_class_classification(cl)
    elif args.testflag:
        helper.test_classification()
    elif args.compute_AUROC:
        helper.compute_AUROC(log_step = args.log_step, epoch_max = args.epochs)
    elif args.plot_training_loss_auroc:
        helper.plot_training_loss_auroc(log_step = args.log_step)
    
    else:
        helper.visualize_latent_vector(args.select)



def parse_arguments():
    # type: () -> Namespace
    """
    Argument parser.

    :return: the command line arguments.
    """
    parser = argparse.ArgumentParser(description = 'Train autoencoder with density estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # autoencoder name 
    parser.add_argument(
        '--autoencoder', 
        type=str,
        help='The Autoencoder framework. Choose among `LSA`,`AAE`', 
        metavar='')
    # density estimator
    parser.add_argument('--estimator', type=str, help='The name of density estimator.'
                        'Choose among `SOS`, `MAF`', metavar='')
    # dataset 
    parser.add_argument('--dataset', type=str,
                        help='The name of the dataset to perform tests on.'
                        'Choose among `mnist`, `cifar10`', metavar='')
    
    parser.add_argument('--PreTrained', dest='pretrained',action = 'store_true',default = False, help = 'Use Pretrained Model')
    parser.add_argument('--Fixed', dest='fixed',action = 'store_true', default = False, help = 'Fix the autoencoder while training')
    parser.add_argument('--MulObj', dest= 'mulobj',action='store_true', default=False)

    parser.add_argument('--Train', dest= 'trainflag',action='store_true', default=False, help = 'Train Mode')
    parser.add_argument('--Test',  dest= 'testflag',action='store_true', default=False, help = 'Test Mode')
    parser.add_argument('--compute_AUROC',  dest= 'compute_AUROC',action='store_true', default=False, help = 'Compute AUROC for trained Models in different epochs')
    parser.add_argument('--plot_training_loss_auroc',  dest= 'plot_training_loss_auroc',action='store_true', default=False, help = 'Plot training loss history and the corresponding AUROC')
    
    # batch size for test
    parser.add_argument(
    '--batch_size',
    type=int,
    default=256,
    help='input batch size for training (default: 100)')
    
    # epochs 
    parser.add_argument(
    '--epochs',
    type=int,
    default=3000,
    help='number of epochs to train/test (default: 3000)')
    

    # epochs before logging 
    parser.add_argument(
    '--before_log_epochs',
    type=int,
    default=30,
    help='number of epochs before logging (default: -1)')

    # select checkpoint
    parser.add_argument(
    '--checkpoint',
    type=str,
    default=None,
    help='number of epochs to check when testing (default: Use the last one)')
    
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
    default=1,
    help='number of invertible blocks (default: 5)')


    # number of blocks
    parser.add_argument(
    '--code_length',
    type=int,
    default=64,
    help='length of hidden vector (default: 64)')

    # number of blocks
    parser.add_argument(
    '--hidden_size',
    type=int,
    default=2048,
    help='length of hidden vector (default: 32)')


    # Normalize the novelty score
    parser.add_argument(
        '--score_normed',
        action ='store_true',
        default= False,
        help ='For Test: Normalize novelty score by Valid Set' )

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

    parser.add_argument(
    '--select_novel_classes',
    '--list',
    nargs='+',
    default = None,
    help = 'Select specific novel classes as test dataset (default: all other classes in dataset)')


    parser.add_argument(
    '--lam',
    type=float,
    default=1,
    help='trade off between reconstruction loss and autoregression loss')
    
    parser.add_argument(
    '--seed',
    type=int,
    default=1,
    help='random_seed')

    parser.add_argument(
    '--log_step',
    type=int,
    default=100,
    help='log_step')

    parser.add_argument(
    '--load_lsa',
    action ='store_true',
    default= False,
    help ='use-pretrained lsa (default: False)' )

    return parser.parse_args()



# Entry point
if __name__ == '__main__':
    main()