#!/bin/bash

echo "START OUR Expermients!"

echo "MAF-LSA CIFAR10"

<<<<<<< e68b04d9643bf8aa75b53953df98d650ab4d948c


# Combine density#
echo "Train"
python train.py --autoencoder LSA --estimator MAF --dataset cifar10 --batch_size 1000 --lr 0.001 --num_blocks 5 --epochs 1000 --combine_density True 


echo "Test score_normed combine_density"
# novel from 0.1 to 0.5
for i in 0.1 0.2 0.3 0.4 0.5 1
# for i in 1 
	do 
		echo "Test on novel_ratio=$i" 
		python test.py --autoencoder LSA --estimator MAF --dataset cifar10 --num_blocks 5 --combine_density True --score_normed True  --novel_ratio $i 
	done


#Not combine density#
echo "Train"
python train.py --autoencoder LSA --estimator MAF --dataset cifar10 --batch_size 1000 --lr 0.001 --num_blocks 5 --epochs 1000 --combine_density False 


echo "Test not score_normed, combine_density"
for i in 0.1 0.2 0.3 0.4 0.5 1
	do 
		echo "Test on novel_ratio=$i" 
		python test.py --autoencoder LSA --estimator MAF --dataset cifar10 --num_blocks 5 --combine_density True --score_normed False  --novel_ratio $i
	done


echo "Test score_normed, not combine_density"
for i in 0.1 0.2 0.3 0.4 0.5 1
	do 
		echo "Test on novel_ratio=$i" 
		python test.py --autoencoder LSA --estimator MAF --dataset cifar10 --num_blocks 5 --combine_density False --score_normed True  --novel_ratio $i
	done



echo "Test not score_normed, not combine_density"
for i in 0.1 0.2 0.3 0.4 0.5 1
	do 
		echo "Test on novel_ratio=$i" 
		python test.py --autoencoder LSA --estimator MAF --dataset cifar10 --num_blocks 5 --combine_density False --score_normed False  --novel_ratio $i
	done
=======
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
>>>>>>> message
