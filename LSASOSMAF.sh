# normalized score
# not combined density

# valid_loss = reconstruction loss
# batch_size = 128
# before_log_epoch = 100
# num_blocks = 3
# lam = 0.1

# LSA_SOS
python train.py  --autoencoder LSA --estimator SOS --epochs 1000  --dataset mnist --n_class 10 --lr 0.0001 --hidden_size 128 --batch_size 32 --num_blocks 3 --lam 0.1 

python test.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset mnist --score_normed --n_class 10 --hidden_size 128 --num_blocks 3 

python train.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset cifar --n_class 10 --lr 0.0001 --hidden_size 128 --batch_size 128 --num_blocks 3 --lam 0.1

python test.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset cifar --score_normed --n_class 10 --hidden_size 128 --num_blocks 3 


#LSA_MAF
python train.py  --autoencoder LSA --estimator MAF --epochs 1000 --dataset mnist --n_class 10 --lr 0.0001 --hidden_size 128 --batch_size 32 --num_blocks 3 --lam 0.1

python test.py  --autoencoder LSA --estimator MAF --epochs 1000 --dataset mnist --score_normed --n_class 10 --hidden_size 128 --num_blocks 3 

python train.py  --autoencoder LSA --estimator MAF --epochs 1000 --dataset cifar --n_class 10 --lr 0.0001 --hidden_size 128 --batch_size 128 --num_blocks 3 --lam 0.1

python test.py  --autoencoder LSA --estimator MAF --epochs 1000 --dataset cifar --score_normed --n_class 10 --hidden_size 128 --num_blocks 3 
