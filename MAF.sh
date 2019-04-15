
python train.py  --NoAutoencoder --estimator MAF --epochs 1000 --dataset mnist --n_class 2 --lr 0.0001  --batch_size 128 --num_blocks 5 --hidden_size 1024

python test.py  --NoAutoencoder --estimator MAF  --dataset mnist --score_normed --n_class 2 --num_blocks 5 --hidden_size 1024

# python test.py  --NoAutoencoder --estimator MAF --epochs 5 --dataset mnist --n_class 2

# python train.py  --NoAutoencoder --estimator MAF --epochs 5 --dataset mnist --Combine_density --n_class 2

# python test.py  --NoAutoencoder --estimator MAF --epochs 5 --dataset mnist --Combine_density --score_normed --n_class 2

# python test.py  --NoAutoencoder --estimator MAF --epochs 5 --dataset mnist --Combine_density --n_class 2


# ##########################################################
# python train.py  --NoAutoencoder --estimator MAF --epochs 5 --dataset cifar10 --n_class 2

# python test.py  --NoAutoencoder --estimator MAF --epochs 5 --dataset cifar10 --score_normed --n_class 2

# python test.py  --NoAutoencoder --estimator MAF --epochs 5 --dataset cifar10  --n_class 2


# python train.py  --NoAutoencoder --estimator MAF --epochs 5 --dataset cifar10 --Combine_density  --n_class 2

# python test.py  --NoAutoencoder --estimator MAF --epochs 5 --dataset cifar10 --Combine_density --score_normed --n_class 2

# python test.py  --NoAutoencoder --estimator MAF --epochs 5 --dataset cifar10 --Combine_density --n_class 2
