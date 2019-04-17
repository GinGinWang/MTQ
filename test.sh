python train.py  --autoencoder LSA --estimator MAF --epochs 10000 --dataset mnist --select 2 --lr 0.0001 --hidden_size 1024 --batch_size 100 --num_blocks 3 --lam 0.01 --before_log_epoch 1000

python test.py  --autoencoder LSA --estimator MAF --dataset mnist --score_normed --select 2 --hidden_size 1024 --num_blocks 3 --batch_size 100