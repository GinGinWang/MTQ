
# python train.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset mnist --n_class 10

# python test.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset mnist --score_normed --n_class 10

# python test.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset mnist --n_class 10

# python train.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset mnist --Combine_density --n_class 10

# python test.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset mnist --Combine_density --score_normed --n_class 10

# python test.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset mnist --Combine_density --n_class 10


##########################################################
# python train.py  --autoencoder LSA --estimator SOS --epochs 10 --dataset cifar10 --n_class 2

# python test.py  --autoencoder LSA --estimator SOS --epochs 10 --dataset cifar10 --score_normed --n_class 2

# python test.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset cifar10  --n_class 10


python train.py  --autoencoder LSA --estimator SOS --epochs 10 --dataset cifar10 --Combine_density  --n_class 2

python test.py  --autoencoder LSA --estimator SOS --epochs 10 --dataset cifar10 --Combine_density --score_normed --n_class 

# python test.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset cifar10 --Combine_density --n_class 10
