import argparse
from argparse import Namespace

from datasets import MNIST

from models import SOSLSA_MNIST
from models import LSAMNIST
from models import SOSAE_MNIST

from result_helpers import OneClassResultHelper

from datasets.utils import set_random_seed

def test_mnist():
    # type: () -> None
    """
    Performs One-class classification tests on MNIST
    """

    # dataset split to train and test
    dataset = MNIST(path='data/MNIST')

    # set model for test
    # model = SOSAE_MNIST(input_shape=dataset.shape, code_length=32, num_blocks= 5).cuda().eval()

    # code_lenghth: latent vector length, cpd_channels:cpd_length
    # model = LSAMNIST(input_shape=dataset.shape, code_length=32, cpd_channels=100).cuda().eval()
    
    model =SOSLSA_MNIST(input_shape=dataset.shape,code_length=32, num_blocks=5).cuda().eval()

    # Set up result helper and perform test
    model_path = 'checkpoints/mnist/'
    helper = OneClassResultHelper(dataset, model, checkpoints_dir=model_path, output_file='mnist_.'+model.name+'txt')
    
    helper.test_one_class_classification()


def parse_arguments():
    # type: () -> Namespace
    """
    Argument parser.

    :return: the command line arguments.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
        test_mnist()
    elif args.dataset == 'cifar10':
        test_cifar()
    else:
        raise ValueError('Unknown dataset')


# Entry point
if __name__ == '__main__':
    main()
