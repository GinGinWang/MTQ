# python train.py  --autoencoder LSA --estimator MAF --epochs 1000 --dataset cifar10 --n_class 10 --hidden_size 128 --batch_size 128 --lr 0.0001 

### lr can not be too large, valid loss will vary a lot
### code_length can not be large, that needs too large hidden_size
### 
# python test.py  --autoencoder LSA --estimator MAF --epochs 1000 --dataset cifar10 --score_normed --n_class 10 --hidden_size 128

# python test.py  --autoencoder LSA --estimator MAF --epochs 1000 --dataset cifar10 --n_class 10 --hidden_size 128

####not do

# python train.py  --autoencoder LSA --estimator MAF --epochs 1000 --dataset cifar10 --Combine_density --n_class 2 --hidden_size 128 --batch_size 128 --lr 0.0001 

# python test.py  --autoencoder LSA --estimator MAF --epochs 1000 --dataset cifar10 --Combine_density --score_normed --n_class 2 --hidden_size 128 

# python test.py  --autoencoder LSA --estimator MAF --epochs 1000 --dataset cifar10 --Combine_density --n_class 10 --hidden_size 128
