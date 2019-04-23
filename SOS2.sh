#!/bin/bash
# START_TIME=$(date +%s.%N)

# python train.py  --autoencoder LSA --estimator SOS --epochs 2000 --dataset mnist  --lr 0.1 --hidden_size 2048 --batch_size 256 --num_blocks 5  --code_length 64  --PreTrained --select 3

# python train.py  --autoencoder LSA --estimator SOS --epochs 2000 --dataset mnist  --lr 0.1 --hidden_size 2048 --batch_size 256 --num_blocks 5  --code_length 64  --PreTrained --select 4

# python train.py  --autoencoder LSA --estimator SOS --epochs 2000 --dataset mnist  --lr 0.1 --hidden_size 2048 --batch_size 256 --num_blocks 5  --code_length 64  --PreTrained --select 5

# python train.py  --autoencoder LSA --estimator SOS --epochs 2000 --dataset mnist  --lr 0.05 --hidden_size 2048 --batch_size 256 --num_blocks 5  --code_length 64  --PreTrained --select 6

# python train.py  --autoencoder LSA --estimator SOS --epochs 2000 --dataset mnist  --lr 0.1 --hidden_size 2048 --batch_size 256 --num_blocks 5  --code_length 64  --PreTrained --select 7

# python train.py  --autoencoder LSA --estimator SOS --epochs 2000 --dataset mnist  --lr 0.1 --hidden_size 2048 --batch_size 256 --num_blocks 5  --code_length 64  --PreTrained --select 8

# python train.py  --autoencoder LSA --estimator SOS --epochs 2000 --dataset mnist  --lr 0.05 --hidden_size 2048 --batch_size 256 --num_blocks 5  --code_length 64  --PreTrained --select 9

# END_TIME=$(date +%s.%N)
# ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
# echo "Runtime: $ELAPSED_TIME seconds"

python test.py  --autoencoder LSA --estimator SOS --dataset mnist   --hidden_size 2048 --num_blocks 10 --batch_size 256 --code_length 64 --score_normed --PreTrained 