import argparse
from argparse import Namespace

from datasets.mnist import MNIST
from datasets.cifar10 import CIFAR10
from datasets.fmnist import FMNIST
from datasets.thyroid import THYROID
from datasets.KDDCUP import KDDCUP

# LSA
from models import LSA_MNIST
from models import LSA_CIFAR10
from models import LSAET_CIFAR10
from models import LSAET_MNIST
from models import AAE_CIFAR10
from models import LSA_KDDCUP
from models import VAE
# Estimator
from models.estimator_1D import Estimator1D
from models.transform_maf import TinvMAF
from models.transform_sos import TinvSOS

from datasets.utils import set_random_seed
from result_helpers import OneClassTestHelper
import os

import torch
import numpy as np

def create_dir(dataset, cd, pretrained, fixed, num_blocks,hidden_size, estimator,noise = 0):
    
    if pretrained:
        dirName = f'checkpoints/{dataset}/combined{cd}/Ptr{pretrained}/Fix{fixed}/'
    else:
        dirName = f'checkpoints/{dataset}/combined{cd}/Ptr{pretrained}/'
    if estimator == 'SOS':
        dirName = f'{dirName}b{num_blocks}h{hidden_size}/'
    if noise >0:
        dirName = f'checkpoints/noise_{noise}/'
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print(f'Make Dir:{dirName}')

    return dirName

def create_file_dir(mulobj,model_name,dataset,cd,pretrained,fixed,score_normed,novel_ratio,num_blocks,hidden_size,lam, add,checkpoint):

    if mulobj:
        dirName = f"results/Mul_{model_name}_{dataset}_cd{cd}_ptr{pretrained}_fix{fixed}_nml{score_normed}_nlr{novel_ratio}_b{num_blocks}_h{hidden_size}_lam{lam}_add{add}"
    else:
       dirName = f"results/{model_name}_{dataset}_cd{cd}_ptr{pretrained}_fix{fixed}_nml{score_normed}_nlr{novel_ratio}_b{num_blocks}_h{hidden_size}_lam{lam}_add{add}"
    if (checkpoint == None):
        dirName= f"{dirName}.txt"
    else:
        dirName = f"{dirName}_at{checkpoint}.txt"
        
    return dirName 

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
    elif args.dataset == 'fmnist':
        dataset = FMNIST(path='data/FMNIST', n_class = args.n_class, select = args.select)

    elif args.dataset == 'cifar10':
        dataset = CIFAR10(path='data/CIFAR', n_class = args.n_class, select= args.select)

    elif args.dataset == 'thyroid':
        dataset = THYROID(path ='data/UCI/thyroid.mat')
    elif args.dataset == 'kddcup':
        dataset = KDDCUP()
    else:
        raise ValueError('Unknown dataset')
    
    
    print ("dataset shape: ",dataset.shape)

    dirName = create_dir(args.dataset, args.cd,args.pretrained, args.fixed,args.num_blocks,args.hidden_size,args.estimator,args.noise)
    
    # c,h,w = dataset.shape

    # Build Model
    # Build Model

    if (args.autoencoder == None):
        # directly estimate density by estimator
        print (f'No Autoencoder, only use Density Estimator: {args.estimator}')
        c,h,w = dataset.shape

        # build Density Estimator
        if args.estimator == 'MAF':
            model = TinvMAF(args.num_blocks, c*h*w,args.hidden_size).cuda()

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
        

    else:
        if args.autoencoder == "LSA":
            print(f'Autoencoder:{args.autoencoder}')
            print(f'Density Estimator:{args.estimator}')
            
            if args.dataset in ['mnist','fmnist']: 

                model =LSA_MNIST(input_shape=dataset.shape, code_length=args.code_length, num_blocks=args.num_blocks, est_name = args.estimator, combine_density = args.cd,hidden_size= args.hidden_size).cuda()
            

            elif args.dataset in ['kddcup']:
                model =LSA_KDDCUP(num_blocks=args.num_blocks, hidden_size= args.hidden_size).cuda()

            elif args.dataset == 'cifar10':
                model =LSA_CIFAR10(input_shape=dataset.shape, code_length=args.code_length, num_blocks=args.num_blocks, est_name= args.estimator, combine_density = args.cd,hidden_size= args.hidden_size).cuda()
            else:
                ValueError("Unknown Dataset")
        
        # elif args.autoencoder == 'LSA_ET':

        #     if args.dataset == 'mnist': 

        #         model =LSAET_MNIST(input_shape=dataset.shape, code_length=args.code_length, num_blocks=args.num_blocks, est_name = args.estimator,hidden_size= args.hidden_size).cuda()
            
        #     elif args.dataset == 'cifar10':
            
        #         model =LSAET_CIFAR10(input_shape=dataset.shape, code_length=args.code_length, num_blocks=args.num_blocks, est_name= args.estimator,hidden_size= args.hidden_size).cuda()
            
        elif args.autoencoder == 'AAE':
            if args.dataset == 'cifar10':
                model =AAE_CIFAR10(input_shape=dataset.shape, code_length=args.code_length, num_blocks=args.num_blocks, est_name= args.estimator, combine_density = args.cd,hidden_size= args.hidden_size).cuda()
            # elif args.dataset == 'mnist':
            # elfi args.dataset == 'fmmnist':
            else:
                ValueError ('Unknown Dataset')

        elif args.autoencoder == 'VAE':

            model = VAE(
            label=args.dataset,
            input_shape = dataset.shape,
            kernel_num= 128,
            z_size = 64,
            ).cuda()
        else:
            raise ValueError('Unknown Autoencoder')
    
    # # set to Test mode
    file_dirName = create_file_dir(args.mulobj,model.name,args.dataset,args.cd,args.pretrained,args.fixed,args.score_normed,args.novel_ratio,args.num_blocks,args.hidden_size,args.lam, args.add, args.checkpoint)

    helper = OneClassTestHelper(dataset, model, args.score_normed, args.novel_ratio, lam = args.lam, checkpoints_dir= dirName, output_file= file_dirName,device = device, batch_size = args.batch_size, pretrained= args.pretrained, trainflag= args.trainflag, lr = args.lr, epochs=args.epochs, before_log_epochs = args.before_log_epochs,pretrained_model= args.premodel,fixed=args.fixed, mulobj=args.mulobj, add = args.add, quantile_flag= args.qt,checkpoint = args.checkpoint, noise= args.noise)

    helper.test_one_class_classification()
    
    
    # if model.name in ['LSA_MAF','LSA_EN','LSA_SOS']:
    #     from matplotlib import pyplot as plt
    #     if args.select == None:
    #         classes = range(0,args.n_class)
    #     else:
    #         classes = [args.select]
        
    #     for cl in classes:
    #     # load result
    #         history_dir = f'{dirName}{cl}{model.name}_history.npy'
    #         history_data = np.load(history_dir).item()
    #         # plot result
    #         epoch_num = len(history_data['val_loss'])
    #         epoch = np.arange(0,epoch_num,1)
            
    #         ax1 = plt.subplot(611)
    #         ax1.plot(epoch, history_data['val_rec'],label = 'val_rec')
    #         ax1.legend(loc = 1)
            
    #         ax2 = plt.subplot(612)
    #         ax2.plot(epoch, history_data['val_nllk'],label = 'val_nllk')
    #         ax2.legend(loc = 1)
            
    #         ax3 = plt.subplot(613)
    #         ax3.plot(epoch, history_data['trn_rec'],label = 'trn_rec')
    #         ax3.legend(loc = 1)

    #         ax4 = plt.subplot(614)
    #         ax4.plot(epoch, history_data['trn_nllk'],label = 'trn_nllk')
    #         ax4.legend(loc = 1)
            
    #         ax5 = plt.subplot(615)
    #         ax5.plot(epoch, history_data['val_loss'],label = 'val_loss')

    #         ax6 = plt.subplot(616)
    #         ax6.plot(epoch, history_data['trn_loss'],label = 'trn_loss')
            
    #         plt.savefig(f'history_image/{cl}{model.name}_{args.dataset}_cd{args.cd}_ptr{args.pretrained}_fix{args.fixed}_nml{args.score_normed}_nlration{args.novel_ratio}_b{args.num_blocks}_h{args.hidden_size}_lam{args.lam}.png')
    #         plt.close('all')
    

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
                        'Choose among `LSA`,`AAE`', metavar='')
    # density estimator
    parser.add_argument('--estimator', type=str, help='The name of density estimator.'
                        'Choose among `SOS`, `MAF`', metavar='')
    # dataset 
    parser.add_argument('--dataset', type=str,
                        help='The name of the dataset to perform tests on.'
                        'Choose among `mnist`, `cifar10`', metavar='')
    
    parser.add_argument('--Combine_density', dest='cd',action = 'store_true',default = False, help = 'Use reconstruction loss as one dimension of latent vector')
    parser.add_argument('--PreTrained', dest='pretrained',action = 'store_true',default = False, help = 'Use Pretrained Model')
    parser.add_argument('--Add', dest='add',action = 'store_true',default = False)
    parser.add_argument('--Qt', dest='qt',action = 'store_true',default = False, help ='Output Quantile-based novelty score')


    parser.add_argument('--premodel',type= str, default='LSA',help = 'Pretrained autoencoder')
    parser.add_argument('--Fixed', dest='fixed',action = 'store_true', default = False, help = 'Fix the autoencoder while training')
    # parser.add_argument('--from_pretrained', dest= 'from_pretrained',action='store_true', default=False)
    parser.add_argument('--NoTrain', dest= 'trainflag',action='store_false', default=True, help = 'Test Mode')

    parser.add_argument('--MulObj', dest= 'mulobj',action='store_true', default=False)
    
    # parser.add_argument('--NoDecoder', dest='decoder_flag',action='store_false', default = True)
    # parser.add_argument('--NoAutoencoder', dest='coder',action='store_false', default = True)
    
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
    default=1000,
    help='number of epochs to train/test (default: 1000)')
    
    # epochs before logging 
    parser.add_argument(
    '--before_log_epochs',
    type=int,
    default=-1,
    help='number of epochs before logging (default: 100)')

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

    parser.add_argument(
    '--lam',
    type=float,
    default=1,
    help='tradeoff between reconstruction loss and autoregression loss')
    
    parser.add_argument(
    '--noise',
    type=float,
    default=0,
    help='noise in training data')

    return parser.parse_args()

# Entry point
if __name__ == '__main__':
    main()
