#!/bin/bash
START_TIME=$(date +%s.%N)

 python train.py  --autoencoder LSA --estimator MAF --epochs 10000 --dataset mnist  --lr 0.00001 --hidden_size 512 --batch_size 256 --num_blocks 5  --code_length 64  --PreTrained --select 4

END_TIME=$(date +%s.%N)
ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
echo "Runtime: $ELAPSED_TIME seconds"

# python train.py  --autoencoder LSA --estimator MAF --epochs 10000 --dataset mnist  --lr 0.000001 --hidden_size 512 --batch_size 256 --num_blocks 5  --code_length 64  --PreTrained

# seperate adjusting parameters 

python test.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 512 --num_blocks 5 --batch_size 256 --code_length 64 --score_normed --select 4

