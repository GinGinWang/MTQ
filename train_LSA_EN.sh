# Train LSA_EN  (to do )
python train_temp.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset mnist  --lr 0.0001 --batch_size 256 --code_length 64 --PreTrained --Fixed

python test_temp.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset mnist --batch_size 256 --code_length 64 --score_normed

python train_temp.py  --autoencoder LSA  --estimator EN --epochs 1000 --dataset cifar10  --lr 0.001 --batch_size 256 --code_length 64 --lam 0.1 --PreTrained --Fixed

python test_temp.py  --autoencoder LSA --estimator EN  --epochs 1000 --dataset cifar10 --batch_size 256 --code_length 64 --score_normed 

# Train LSA_EN 
python train_temp.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset mnist  --lr 0.0001 --batch_size 256 --code_length 64 --PreTrained

python test_temp.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset mnist --batch_size 256 --code_length 64 --score_normed

python train_temp.py  --autoencoder LSA  --estimator EN --epochs 1000 --dataset cifar10  --lr 0.001 --batch_size 256 --code_length 64 --lam 0.1 --PreTrained

python test_temp.py  --autoencoder LSA --estimator EN  --epochs 1000 --dataset cifar10 --batch_size 256 --code_length 64 --score_normed

# Train LSA_EN 
# python train_temp.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset mnist  --lr 0.0001 --batch_size 256 --code_length 64

# python test_temp.py  --autoencoder LSA --estimator EN --epochs 1000 --dataset mnist --batch_size 256 --code_length 64 --score_normed

# python train_temp.py  --autoencoder LSA  --estimator EN --epochs 1000 --dataset cifar10  --lr 0.001 --batch_size 256 --code_length 64 --lam 0.1

# python test_temp.py  --autoencoder LSA --estimator EN  --epochs 1000 --dataset cifar10 --batch_size 256 --code_length 64 --score_normed



