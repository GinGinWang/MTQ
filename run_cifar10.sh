#!/bin/bash

echo "START OUR Expermients!"

echo "MAF-LSA CIFAR10"

# Combine density#
echo "Train"
python train.py  --autoencoder LSA --estimator MAF --dataset cifar10  --num_blocks 5 --code_length 32         

# echo "Test score_normed combine_density"

# for i in 0.1 0.2 0.3 0.4 0.5 1
# for i in 1 
	# do 
		echo "Test on novel_ratio=$i" 
		python test.py --autoencoder LSA --estimator MAF --dataset cifar10 --num_blocks 5 --code_length 32 
	# done


# combine density
# echo "Train"
# python train.py  --autoencoder LSA --estimator MAF --dataset cifar10  --num_blocks 5 --epochs 100 --code_length 32 --Combine_density       

# 	do 
# 		echo "Test on novel_ratio=$i" 
# 		python test.py --autoencoder LSA --estimator MAF --dataset cifar10 --num_blocks 5 --code_length 32 --Combine_density
# 	done


# # MAF

# echo "Train"
# python train.py  --estimator MAF --dataset cifar10  --num_blocks 5 --epochs 100 --code_length 32 --NoAutoencoder       

# 	do 
# 		echo "Test on novel_ratio=$i" 
# 		python test.py --estimator MAF --dataset cifar10 --num_blocks 5 --code_length 32 --NoAutoencoder
# 	done


# echo "Train"
# python train.py  --estimator MAF --dataset cifar10  --num_blocks 5 --epochs 100 --code_length 32 --NoAutoencoder --Combine_density      

# 	do 
# 		echo "Test on novel_ratio=$i" 
# 		python test.py --estimator MAF --dataset cifar10 --num_blocks 5 --code_length 32 --NoAutoencoder --Combine_density
# 	done
