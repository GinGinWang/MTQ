#!/bin/bash

echo "START OUR Expermients!"

echo "SOSLSA MINIST"

# echo "Train"
# python train.py --autoencoder LSA --estimator MAF --dataset mnist --batch-size 100 --lr 0.001 --num-blocks 5  --epochs 1

echo "Test"
python test.py --autoencoder LSA --estimator MAF --dataset mnist --batch-size 1 --lr 0.001 --num-blocks 5  --epochs 1