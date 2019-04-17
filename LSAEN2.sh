
python train.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset mnist --n_class 10 --lr 0.01 --batch_size 32 --PreTrained

python test.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset mnist --score_normed --n_class 10 --PreTrained

python test.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset mnist --n_class 10 --PreTrained


python train.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset mnist --Combine_density --n_class 10 --lr 0.01 --batch_size 32 --PreTrained

python test.py  --autoencoder LSA --estimator EN --epochs 2000 --dataset mnist --Combine_density --score_normed --n_class 10  --PreTrained

python test.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset mnist --Combine_density   --n_class 10  --PreTrained


# ##################################################
python train.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset cifar10 --n_class 10 --lr 0.001 --batch_size 128 --PreTrained

python test.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset cifar10 --score_normed --n_class 10 --PreTrained

python test.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset cifar10 --n_class 10 --PreTrained


python train.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset cifar10 --Combine_density --lr 0.001 --batch_size 128 --PreTrained

python test.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset cifar10 --Combine_density --score_normed --n_class 10 --PreTrained

python test.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset cifar10 --Combine_density  --n_class 10 --PreTrained
