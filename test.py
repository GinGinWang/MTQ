"""

Multivariate Triangular Map For Novelty Detection.

By Jing Jing Wang/ 2019.10

"""
import argparse
from argparse import Namespace
import os
import torch
import numpy as np
import torch.nn as nn

from datasets import *

# models
# auto-encoder
from models.LSA_mnist import LSA_MNIST
from models import LSA_KDDCUP
# density estimator
from models.estimator_1D import Estimator1D
# MTQ
from models.transform_maf import TinvMAF
from models.transform_sos import TinvSOS

# random seed
from utils import set_random_seed
# helper class for training/testing/plotting
from result_helpers.test_one_class import OneClassTestHelper
# path
from utils import create_checkpoints_dir
from utils import create_file_path


def main():
    """
    Main Function.

    Training/Test/Plot
    """
    args = parse_arguments()
    device = torch.device('cuda')

    # remove randomness
    set_random_seed(args.seed)

    # Set Dataset
    if args.dataset == 'mnist':
        dataset = MNIST(path='data/MNIST',
                        n_class=args.n_class,
                        select=args.select,
                        select_novel_classes=args.select_novel_classes)

    elif args.dataset == 'fmnist':
        dataset = FMNIST(path='data/FMNIST',
                         n_class=args.n_class,
                         select=args.select)

    elif args.dataset == 'thyroid':
        dataset = THYROID(path='data/UCI')

    elif args.dataset == 'kddcup':
        dataset = KDDCUP(path='data/UCI')
    else:
        raise ValueError('Unknown dataset')

    checkpoints_dir = create_checkpoints_dir(
        args.dataset, args.fixed, args.mulobj,
        args.num_blocks, args.hidden_size,
        args.code_length, args.estimator)

    # Set Model
    if (args.autoencoder is None):
        print (f'No Autoencoder, only use Density Estimator: {args.estimator}')
        c, h, w = dataset.shape

        # build Density Estimator
        if args.estimator == 'SOS':
            model = TinvSOS(args.num_blocks,
                            c * h * w, args.hidden_size).cuda()
        # 1-D estimator from LSA
        elif args.estimator == 'EN':
            self.model = Estimator1D(
                code_length=c * h * w,
                fm_list=[32, 32, 32, 32],
                cpd_channels=100).cuda()
        else:
            raise ValueError('Unknown Estimator')
    else:
        if args.autoencoder == "LSA":
            print(f'Autoencoder:{args.autoencoder}')
            print(f'Density Estimator:{args.estimator}')
            if args.dataset in ['mnist', 'fmnist']:
                model = LSA_MNIST(
                    input_shape=dataset.shape,
                    code_length=args.code_length,
                    num_blocks=args.num_blocks,
                    est_name=args.estimator,
                    hidden_size=args.hidden_size).cuda()

            elif args.dataset in ['kddcup']:
                model = LSA_KDDCUP(
                    num_blocks=args.num_blocks,
                    hidden_size=args.hidden_size,
                    code_length=args.code_length,
                    est_name=args.estimator).cuda()

            elif args.dataset in ['thyroid']:
                model = LSA_THYROID(
                    num_blocks=args.num_blocks,
                    hidden_size=args.hidden_size,
                    code_length=args.code_length,
                    est_name=args.estimator).cuda()
            else:
                ValueError("Unknown Dataset")
        else:
            raise ValueError('Unknown Autoencoder')

    # Result saved path
    result_file_path = create_file_path(
        args.mulobj, args.fixed, args.pretrained,
        model.name, args.dataset, args.score_normed,
        args.num_blocks, args.hidden_size,
        args.code_length, args.lam, args.checkpoint)

    print(checkpoints_dir)

    helper = OneClassTestHelper(
        dataset=dataset,
        model=model,
        score_normed=args.score_normed,
        lam=args.lam,
        checkpoints_dir=checkpoints_dir,
        result_file_path=result_file_path,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        before_log_epochs=args.before_log_epochs,
        code_length=args.code_length,
        mulobj=args.mulobj,
        test_checkpoint=args.checkpoint,
        log_step=args.log_step,
        device=device,
        fixed=args.fixed,
        pretrained=args.pretrained,
        load_lsa=args.load_lsa)

    if args.trainflag:
        cl = args.select
        helper.train_one_class_classification(cl)
    elif args.testflag:
        helper.test_classification()
    elif args.compute_AUROC:
        helper.compute_AUROC(log_step=args.log_step, epoch_max=args.epochs)
    elif args.plot_training_loss_auroc:
        helper.plot_training_loss_auroc(log_step=args.log_step)
    elif args.using_train_set:
        cl = args.select
        helper.test_one_class_classification_with_trainset(cl)
    # else:
    #     helper.visualize_latent_vector(args.select)


def parse_arguments():
    """

    Argument parser.

    :return: the command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Train autoencoder with TQM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--autoencoder',
        type=str,
        help='The Autoencoder framework. Choose among `LSA`,`AAE`',
        metavar='')

    # density estimator
    parser.add_argument(
        '--estimator',
        type=str,
        help='The name of density estimator/TQM.\
        Choose among `SOS`, `MAF` and `EN`', metavar='')

    # Dataset
    parser.add_argument(
        '--dataset', type=str,
        help='The name of the dataset to perform tests on.'
        'Choose among `mnist`, `fmnist`,\
        `thyroid`,`kddcup`', metavar='')

    # Training Strategy
    parser.add_argument(
        '--PreTrained',
        dest='pretrained',
        action='store_true',
        default=False,
        help='Use Pretrained Model')
    parser.add_argument(
        '--Fixed',
        dest='fixed',
        action='store_true',
        default=False,
        help='Fix the autoencoder while training')
    parser.add_argument(
        '--MulObj',
        dest='mulobj',
        action='store_true',
        default=False)

    # Setting model mode (Train or Test)
    parser.add_argument(
        '--Train',
        dest='trainflag',
        action='store_true',
        default=False,
        help='Train Mode')
    parser.add_argument(
        '--Test',
        dest='testflag',
        action='store_true',
        default=False,
        help='Test Mode')
    parser.add_argument(
        '--compute_AUROC',
        dest='compute_AUROC',
        action='store_true',
        default=False,
        help='Compute AUROC for trained Models in different epochs')

    # batch size for test
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='input batch size for training (default: 100)')

    # Maximum epochs  for training
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

    # select specific checkpoint when test
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='number of epochs to check when testing\
        (default: Use the last one)')

    # learning rate
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='learning rate (default: 0.0001)')

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
        default=1,
        help='number of invertible blocks (default: 5)')

    # length of latent vector
    parser.add_argument(
        '--code_length',
        type=int,
        default=64,
        help='length of hidden vector (default: 64)')

    # hidden size of conditioner network
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=2048,
        help='length of hidden vector (default: 32)')

    # Normalize the novelty score
    parser.add_argument(
        '--score_normed',
        action='store_true',
        default=False,
        help='For Test: Normalize novelty score by Valid Set')

    parser.add_argument(
        '--n_class',
        type=int,
        default=10,
        help='Number of classes used in experiments')

    parser.add_argument(
        '--select',
        type=int,
        default=None,
        help='Select one specific class (as nominal class) for training')

    parser.add_argument(
        '--select_novel_classes',
        '--list',
        nargs='+',
        default=None,
        help='Select specific novel classes in test dataset (default: using all\
         other classes in dataset as novel class)')

    parser.add_argument(
        '--lam',
        type=float,
        default=1,
        help='trade off between reconstruction loss and auto-regression loss')

    # change random seed
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='random_seed')

    parser.add_argument(
        '--log_step',
        type=int,
        default=100,
        help='log_step, save model for every #log_step epochs')

    # Load saved autoecoder as the start of training
    parser.add_argument(
        '--load_lsa',
        action='store_true',
        default=False,
        help='use-pretrained lsa (default: False)')

    parser.add_argument(
        '--using_train_set',
        action='store_true',
        default=False,
        help='use training set on saved model in Test mode (default: False)')

    # Additional function for analyzing results
    parser.add_argument(
        '--plot_training_loss_auroc',
        dest='plot_training_loss_auroc',
        action='store_true',
        default=False,
        help='Plot training loss history and the corresponding AUROC')

    return parser.parse_args()


if __name__ == '__main__':
    """
    entry point.
    """

    main()
