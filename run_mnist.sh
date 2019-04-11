#!/bin/bash

echo "START OUR Expermients!"

echo "MAF-LSA MINIST"

<<<<<<< e68b04d9643bf8aa75b53953df98d650ab4d948c
# Not combine density#
echo "Train"
python train.py --autoencoder LSA --estimator MAF --dataset mnist --batch_size 1000 --lr 0.001 --num_blocks 5 --epochs 1000 --combine_density False  --code_length 32

# Combine density#
echo "Train"
python train.py --autoencoder LSA --estimator MAF --dataset mnist --batch_size 1000 --lr 0.001 --num_blocks 5 --epochs 1000 --combine_density True --code_length 32


# echo "Test score_normed, not combine_density"
for i in  0.1 0.2 0.3 0.4 0.5 1 
# for i in 1
	do 
		echo "Test on novel_ratio=$i" 
		python test.py --autoencoder LSA --estimator MAF --dataset mnist --num_blocks 5 --combine_density False --score_normed True  --novel_ratio $i --code_length 32
=======
Not combine density# done
echo "Train"
python train.py  --autoencoder LSA --estimator MAF --dataset mnist  --num_blocks 6 --code_length 10

echo "Test score_normed, not combine_density"
for i in 1 
# for i in 0.1
	do 
		echo "Test on novel_ratio=$i" 
		python test.py --autoencoder LSA --estimator MAF --dataset mnist --num_blocks 6 --code_length 10 
>>>>>>> message
	done




<<<<<<< e68b04d9643bf8aa75b53953df98d650ab4d948c


echo "Test score_normed combine_density"
# novel from 0.1 to 0.5
for i in  0.1 0.2 0.3 0.4 0.5 1
# for i in  0.1
	do 
		echo "Test on novel_ratio=$i" 
		python test.py --autoencoder LSA --estimator MAF --dataset mnist --num_blocks 5 --combine_density True --score_normed True  --novel_ratio $i --code_length 32 
	done


echo "Test not score_normed, combine_density"
for i in  0.1 0.2 0.3 0.4 0.5 1 
	do 
		echo "Test on novel_ratio=$i" 
		python test.py --autoencoder LSA --estimator MAF --dataset mnist --num_blocks 5 --combine_density True --score_normed False  --novel_ratio $i --code_length 32
	done


=======
# # Combine density# nan
# echo "Train"
# python train.py --autoencoder LSA --estimator MAF --dataset mnist --batch_size 1000  --num_blocks 5 --epochs 1000 --combine_density True --code_length 32


#! doing


# echo "Test score_normed combine_density"
# # novel from 0.1 to 0.5
# for i in  0.1 0.2 0.3 0.4 0.5 1
# # for i in  0.1
# 	do 
# 		echo "Test on novel_ratio=$i" 
# 		python test.py --autoencoder LSA --estimator MAF --dataset mnist --num_blocks 5 --combine_density True --score_normed True  --novel_ratio $i --code_length 32 
# 	done


# echo "Test not score_normed, combine_density"
# for i in  0.1 0.2 0.3 0.4 0.5 1 
# 	do 
# 		echo "Test on novel_ratio=$i" 
# 		python test.py --autoencoder LSA --estimator MAF --dataset mnist --num_blocks 5 --combine_density True --score_normed False  --novel_ratio $i --code_length 32
# 	done
>>>>>>> message




<<<<<<< e68b04d9643bf8aa75b53953df98d650ab4d948c
echo "Test not score_normed, not combine_density"
for i in  0.1 0.2 0.3 0.4 0.5 1 
	do 
		echo "Test on novel_ratio=$i" 
		python test.py --autoencoder LSA --estimator MAF --dataset mnist --num_blocks 5 --combine_density False --score_normed False  --novel_ratio $i --code_length 32
	done
=======


# echo "Test not score_normed, not combine_density"
# for i in  0.1 0.2 0.3 0.4 0.5 1 
# 	do 
# 		echo "Test on novel_ratio=$i" 
# 		python test.py --autoencoder LSA --estimator MAF --dataset mnist --num_blocks 5 --combine_density False --score_normed False  --novel_ratio $i --code_length 32
# 	done
>>>>>>> message
