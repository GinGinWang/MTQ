
# python train.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset mnist --n_class 10 --lr 0.01

# python test.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset mnist --score_normed --n_class 10 

# python test.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset mnist --n_class 10 


# python train.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset mnist --Combine_density --n_class 10 --lr 0.01

# python test.py  --autoencoder LSA --estimator EN --epochs 2000 --dataset mnist --Combine_density --score_normed --n_class 10  

# python test.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset mnist --Combine_density   --n_class 10  


##################################################
python train.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset cifar10 --n_class 2 --lr 0.01

python test.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset cifar10 --score_normed --n_class 2

# python test.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset cifar10 --n_class 10


# python train.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset cifar10 --Combine_density 10

# python test.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset cifar10 --Combine_density --score_normed --n_class 10

# python test.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset cifar10 --Combine_density  --n_class 10
