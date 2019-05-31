import numpy as np
import os 
import argparse
from argparse import Namespace

def main():
	args = parse_arguments()
	dirpath = os.getcwd()
	result_path = f"{dirpath}/results/{args.dataset}/"

	listing = os.listdir(result_path)t    
	i = 0
	ave_AUROC = 0
	for infile in listing:
		if infile.endswith("npz") and infile.startswith(f"cifar10_{args.algorithm}_"):
			data = np.load(f"{result_path}{infile}")
			print (f"class-{i}: roc_auc:{data['roc_auc']}")
			ave_AUROC += data['roc_auc']
			i +=1
	ave_AUROC/=float(i)
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