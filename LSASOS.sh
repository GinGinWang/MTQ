
# python train.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset mnist --n_class 10 --lr 0.0001 --hidden_size 128 --batch_size 32 --num_blocks 3 --lam 0.1

# python test.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset mnist --score_normed --n_class 10 --hidden_size 128 --num_blocks 3 

# python test.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset mnist  --n_class 10 --hidden_size 128  --num_blocks 3

# python train.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset mnist --Combine_density --n_class 10 --lr 0.0001 --hidden_size 128 --batch_size 32 --num_blocks 3 --lam 0.1


# python test.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset mnist --Combine_density --score_normed --n_class 10 --hidden_size 128 --num_blocks 3

# python test.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset mnist --Combine_density --n_class 10 --hidden_size 128 --num_blocks 3


##########################################################
# python train.py  --autoencoder LSA --estimator SOS --epochs 10 --dataset cifar10 --n_class 2 --lr 0.0001 --hidden_size 128 --batch_size 32 --num_blocks 3 --lam 0.1

# python test.py  --autoencoder LSA --estimator SOS --epochs 10 --dataset cifar10 --score_normed --n_class 2

# python test.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset cifar10  --n_class 10


# python train.py  --autoencoder LSA --estimator SOS --epochs 10 --dataset cifar10 --Combine_density  --n_class 2

# python test.py  --autoencoder LSA --estimator SOS --epochs 10 --dataset cifar10 --Combine_density --score_normed --n_class 

# python test.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset cifar10 --Combine_density --n_class 10


python train.py  --autoencoder LSA --estimator MAF --epochs 10000 --dataset mnist  --lr 0.0001 --hidden_size 4098 --batch_size 100 --num_blocks 5 --before_log_epoch 200 --PreTrained

python test.py  --autoencoder LSA --estimator MAF --dataset mnist --score_normed  --hidden_size 4098 --num_blocks 5 --batch_size 100 --PreTrained