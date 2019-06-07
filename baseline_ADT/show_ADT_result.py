import numpy as np
import os 
import argparse
from argparse import Namespace
from utils import get_class_name_from_index
def main():
    args = parse_arguments()
    dirpath = os.getcwd()
    result_path = f"{dirpath}/results/{args.dataset}/"

    listing = os.listdir(result_path)    
    ave_AUROC = 0

    for i in range(10):
        class_name = get_class_name_from_index(i, args.dataset)
        for file in listing:
            if file.startswith(f"{args.dataset}_{args.algorithm}_{class_name}") and file.endswith("npz"):

                data =np.load(f"{result_path}{file}")

        print (f"class-{i}: roc_auc:{data['roc_auc']}")
        ave_AUROC += data['roc_auc']

    ave_AUROC/=10.0
    print (f"{args.algorithm}-Average of AUROC:{ave_AUROC}")
        
def parse_arguments():
    # type: () -> Namespace
    """
    Argument parser.

    :return: the command line arguments.
    """
    parser = argparse.ArgumentParser(description = 'Show results from ADT method',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # autoencoder name 
    parser.add_argument('--dataset', type=str,
        help='Dataset Name'
        'Choose among `cifar10`,`mnist`', metavar='')
    parser.add_argument('--algorithm', type=str,
        help='Algorithm Name'
        'Choose among `dsebm`,`transformations`', metavar='')
    
    return parser.parse_args()


if __name__ == '__main__':        
    main()