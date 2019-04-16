
python train.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset cifar10 --n_class 2 --lr 0.1 --hidden_size 256 --batch_size 64 --num_blocks 3 --lam 1 

python test.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset cifar10 --score_normed --n_class 2 --hidden_size 64 --num_blocks 3 

# python test.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset cifar10  --n_class 10 --hidden_size 128  --num_blocks 3

# python train.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset cifar10 --Combine_density --n_class 10 --lr 0.0001 --hidden_size 128 --batch_size 128 --num_blocks 3 --lam 0.1


# python test.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset cifar10 --Combine_density --score_normed --n_class 10 --hidden_size 128 --num_blocks 3

# python test.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset cifar10 --Combine_density --n_class 10 --hidden_size 128 --num_blocks 3

