import argparse
from argparse import Namespace

from datasets.mnist import MNIST
from datasets.cifar10 import CIFAR10
<<<<<<< HEAD
=======
<<<<<<< e68b04d9643bf8aa75b53953df98d650ab4d948c
from models import LSA_MNIST
from models import LSA_CIFAR10

=======
>>>>>>> temp

# LSA
from models import LSA_MNIST
from models import LSA_CIFAR10

# Estimator
from models.estimator_1D import Estimator1D
from models.transform_maf import TinvMAF
from models.transform_sos import TinvSOS

<<<<<<< HEAD
=======
>>>>>>> message
>>>>>>> temp
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
<<<<<<< HEAD
        dataset = MNIST(path='data/MNIST', n_class = args.n_class)
    elif args.dataset == 'cifar10':
        dataset = CIFAR10(path='data/CIFAR10', n_class = args.n_class)
=======
<<<<<<< e68b04d9643bf8aa75b53953df98d650ab4d948c
        dataset = MNIST(path='data/MNIST')
        lam =0.1

    elif args.dataset == 'cifar10':
        dataset = CIFAR10(path='data/CIFAR10')
        lam = 0.1
    
=======
        dataset = MNIST(path='data/MNIST', n_class = args.n_class)
    elif args.dataset == 'cifar10':
        dataset = CIFAR10(path='data/CIFAR10', n_class = args.n_class)
>>>>>>> message
>>>>>>> temp
    else:
        raise ValueError('Unknown dataset')
    
    print ("dataset shape: ",dataset.shape)
<<<<<<< HEAD
=======
<<<<<<< e68b04d9643bf8aa75b53953df98d650ab4d948c
    

    




    # trained model save_dir
    dirName = f'checkpoints/{args.dataset}/combined{args.combine_density}/'
=======
>>>>>>> temp
    c, h , w = dataset.shape
     
    # trained model save_dir
    dirName = f'checkpoints/{args.dataset}/combined{args.cd}/'
<<<<<<< HEAD
=======
>>>>>>> message
>>>>>>> temp
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print(f'Make Dir:{dirName}')



    for cl_idx, cl in enumerate(dataset.train_classes):

        dataset.train(cl)
        # Build Model
<<<<<<< HEAD
=======
<<<<<<< e68b04d9643bf8aa75b53953df98d650ab4d948c
        if args.autoencoder == "LSA":
            
            if args.dataset == 'mnist':        
                model =LSA_MNIST(input_shape=dataset.shape, code_length=args.code_length, num_blocks=args.num_blocks, est_name= args.estimator, combine_density = args.combine_density).cuda()
            
            elif args.dataset == 'cifar10':
                model =LSA_CIFAR10(input_shape=dataset.shape, code_length=args.code_length, num_blocks=args.num_blocks, est_name= args.estimator, combine_density = args.combine_density).cuda()

=======
>>>>>>> temp
        if (not args.coder):
            # directly estimate density by model

            # build Density Estimator
            if args.estimator == 'MAF':
                model = TinvMAF(args.num_blocks, c*h*w).cuda()

            elif args.estimator == 'SOS':
                model = TinvSOS(args.num_blocks, c*h*w).cuda()

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

                    model =LSA_MNIST(input_shape=dataset.shape, code_length=args.code_length, num_blocks=args.num_blocks, est_name= args.estimator, combine_density = args.cd).cuda()
                
                # elif args.dataset == 'cifar10':
                
                #     model =LSA_CIFAR10(input_shape=dataset.shape, code_length=args.code_length, num_blocks=args.num_blocks, est_name= args.estimator, combine_density = args.cd).cuda()
            else:
                raise ValueError('Unknown MODEL')
            
<<<<<<< HEAD
=======
>>>>>>> message
>>>>>>> temp

        # (add other models here)    

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

        

        
        # Initialize training process
<<<<<<< HEAD
        helper = OneClassTrainHelper(dataset, model, optimizer, lam = args.lam,  checkpoints_dir=dirName, train_epoch=args.epochs, batch_size= args.batch_size)
=======
<<<<<<< e68b04d9643bf8aa75b53953df98d650ab4d948c
        helper = OneClassTrainHelper(dataset, model, optimizer, lam = lam, checkpoints_dir=dirName, train_epoch=args.epochs, batch_size= args.batch_size)
=======
        helper = OneClassTrainHelper(dataset, model, optimizer, lam = args.lam,  checkpoints_dir=dirName, train_epoch=args.epochs, batch_size= args.batch_size)
>>>>>>> message
>>>>>>> temp

        # Start training 
        helper.train_one_class_classification()



















<<<<<<< HEAD
=======
<<<<<<< e68b04d9643bf8aa75b53953df98d650ab4d948c
=======
>>>>>>> temp











<<<<<<< HEAD
=======
>>>>>>> message
>>>>>>> temp
def parse_arguments():
    # type: () -> Namespace
    """
    Argument parser.

    :return: the command line arguments.
    """
    parser = argparse.ArgumentParser(description = 'Train autoencoder with density estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

<<<<<<< HEAD
=======
<<<<<<< e68b04d9643bf8aa75b53953df98d650ab4d948c
    # autoencoder name 
    parser.add_argument('--autoencoder', type=str,
                        help='The Autoencoder framework.'
                        'Choose among `LSA`', metavar='')
    # density estimator
    parser.add_argument('--estimator', type=str, default='SOS', help='The name of density estimator.'
=======
>>>>>>> temp

    # autoencoder name 
    parser.add_argument('--autoencoder', type=str,default ='LSA',
                        help='The Autoencoder framework.'
                        'Choose among `LSA`', metavar='')
    # density estimator
    parser.add_argument('--estimator', type=str, default=None, help='The name of density estimator.'
<<<<<<< HEAD
=======
>>>>>>> message
>>>>>>> temp
                        'Choose among `SOS`, `MAF`', metavar='')
    # dataset 
    parser.add_argument('--dataset', type=str,
                        help='The name of the dataset to perform tests on.'
                        'Choose among `mnist`, `cifar10`', metavar='')
<<<<<<< HEAD
=======
<<<<<<< e68b04d9643bf8aa75b53953df98d650ab4d948c
    
=======
>>>>>>> temp

    parser.add_argument('--Combine_density', dest='cd',action = 'store_true',default = False)

    parser.add_argument('--NoAutoencoder', dest='coder',action='store_false', default = True)

<<<<<<< HEAD
=======
>>>>>>> message
>>>>>>> temp
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
<<<<<<< HEAD
=======
<<<<<<< e68b04d9643bf8aa75b53953df98d650ab4d948c
    default=64,
    help='length of hidden vector (default: 32)')

    # join density (For train or test)
    parser.add_argument(
        '--combine_density',
        default= False,
        help = 'Combine reconstruction loss in the input of density estimator'
        )

=======
>>>>>>> temp
    default=32,
    help='length of hidden vector (default: 32)')

    # number of blocks
    parser.add_argument(
    '--lam',
    type=float,
    default=1,
    help='tradeoff between reconstruction loss and autoregression loss')
    
    parser.add_argument(
    '--n_class',
    type = int,
    default = 2,
    help = 'Number of classes used in experiments')
<<<<<<< HEAD
=======
>>>>>>> message
>>>>>>> temp

    #K  (only for SOS flow) 
    #M (only for SOS flow)

    return parser.parse_args()
    
# Entry point
if __name__ == '__main__':
    main()
