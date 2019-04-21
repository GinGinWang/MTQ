#!/bin/bash
# START_TIME=$(date +%s.%N)

# python train.py  --autoencoder LSA --estimator MAF --epochs 10000 --dataset mnist  --lr 0.00001 --hidden_size 512 --batch_size 256 --num_blocks 5  --code_length 64  --PreTrained

# END_TIME=$(date +%s.%N)
# ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
# echo "Runtime: $ELAPSED_TIME seconds"

# python test.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 512 --num_blocks 5 --batch_size 256 --code_length 64 --score_normed --PreTrained 


# START_TIME=$(date +%s.%N)

# python train.py  --autoencoder LSA --estimator MAF --epochs 10000 --dataset cifar10  --lr 0.00001 --hidden_size 512 --batch_size 256 --num_blocks 5  --code_length 64  --PreTrained 

# END_TIME=$(date +%s.%N)
# ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
# echo "Runtime: $ELAPSED_TIME seconds"

# python test.py  --autoencoder LSA --estimator MAF --dataset cifar10   --hidden_size 512 --num_blocks 5 --batch_size 256 --code_length 64 --score_normed --PreTrained


## optimizer for whole model -- load pretrained autoencoder first

START_TIME=$(date +%s.%N)

python train_temp.py  --autoencoder LSA --estimator MAF --epochs 10000 --dataset mnist  --lr 0.000001 --hidden_size 512 --batch_size 256 --num_blocks 5  --code_length 64 --lam 1 --before_log_epochs 30 --PreTrained --select 5

END_TIME=$(date +%s.%N)
ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
echo "Runtime: $ELAPSED_TIME seconds"

python test_temp.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 512 --num_blocks 5 --batch_size 256 --code_length 64 --score_normed --PreTrained


## optimizer for whole model -- from random initialization

START_TIME=$(date +%s.%N)

python train_temp.py  --autoencoder LSA --estimator MAF --epochs 10000 --dataset mnist  --lr 0.000001 --hidden_size 512 --batch_size 256 --num_blocks 5  --code_length 64 --lam 1 --before_log_epochs 30 --PreTrained --select 5

END_TIME=$(date +%s.%N)
ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
echo "Runtime: $ELAPSED_TIME seconds"

python test_temp.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 512 --num_blocks 5 --batch_size 256 --code_length 64 --score_normed --PreTrained
