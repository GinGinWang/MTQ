#!/bin/bash
START_TIME=$(date +%s.%N)

python train.py  --autoencoder LSA --estimator SOS --epochs 2000 --dataset cifar10  --lr 0.1 --hidden_size 2048 --batch_size 256 --num_blocks 5  --code_length 64  --PreTrained 

python test.py  --autoencoder LSA --estimator SOS --dataset cifar10   --hidden_size 2048 --num_blocks 5 --batch_size 256 --code_length 64 --score_normed --PreTrained 

END_TIME=$(date +%s.%N)
ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
echo "Runtime: $ELAPSED_TIME seconds"