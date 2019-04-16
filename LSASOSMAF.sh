# normalized score
# valid_loss = reconstruction loss

## framework
# not combined density
# hidden_size = 128
# code_length = 64
# K = 5 r=3
#  num_blocks = 3

## Training setting
# batch_size = 128
# before_log_epoch = 100
# lam = 0.1


### HOW TO GET GOOD Results!!!!!

# LSA_SOS
python train.py  --autoencoder LSA --estimator SOS --epochs 1000  --dataset mnist --select 2 --lr 0.0001 --hidden_size 4096 --batch_size 32 --num_blocks 5 --lam 0.1 

python test.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset mnist  --select 2 --hidden_size 4096 --num_blocks 5

python test.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset mnist  --select 2 --hidden_size 4096 --num_blocks 5 --score_normed


# python train.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset cifar --n_class 10 --lr 0.0001 --hidden_size 128 --batch_size 128 --num_blocks 3 --lam 0.1

# python test.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset cifar --score_normed --n_class 10 --hidden_size 128 --num_blocks 3 


#LSA_MAF
# python train.py  --autoencoder LSA --estimator MAF --epochs 1000 --dataset mnist --n_class 10 --lr 0.0001 --hidden_size 128 --batch_size 32 --num_blocks 3 --lam 0.1

# python test.py  --autoencoder LSA --estimator MAF --epochs 1000 --dataset mnist --score_normed --n_class 10 --hidden_size 128 --num_blocks 3 

# python train.py  --autoencoder LSA --estimator MAF --epochs 1000 --dataset cifar --n_class 10 --lr 0.0001 --hidden_size 128 --batch_size 128 --num_blocks 3 --lam 0.1

# python test.py  --autoencoder LSA --estimator MAF --epochs 1000 --dataset cifar --score_normed --n_class 10 --hidden_size 128 --num_blocks 3 
