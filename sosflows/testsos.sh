#!/bin/bash
#### environment python3.6

echo "START OUR SOS Expermients!"

echo "POWER"
python main_ss.py --dataset POWER --batch-size 1000 --lr 0.001 --num-blocks 5 --K 4 --M 3 --epochs 1000 --log-interval 100

# echo "POWER_m"
# python main_ss_m.py --dataset POWER --batch-size 1000 --lr 0.001 --num-blocks 5 --K 4 --M 3 --epochs 1000 

echo "GAS"
python main_ss.py --dataset GAS --batch-size 1000 --lr 0.001 --num-blocks 5 --K 4 --M 3 --epochs 1000 --log-interval 100

# echo "GAS_m"
# python main_ss_m.py --dataset GAS --batch-size 1000 --lr 0.001 --num-blocks 5 --K 5 --M 3 --epochs 1000

#echo "HEPMASS"
#python main_ss.py --dataset HEPMASS
# echo "MINIBONE"
# python main_ss.py --dataset MINIBONE
# echo "BSDS300"
# python main_ss.py --dataset BSDS300

# echo "TEST_LOSS:"
# python show_result.py 



# python main_ss.py --dataset POWER --batch-size 100 --lr 0.001 --num-blocks 5 --K 5 --M 3 
#  --epochs 1