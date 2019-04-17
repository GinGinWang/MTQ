python train.py  --autoencoder LSA --estimator MAF --epochs 10000 --dataset mnist  --lr 0.0001 --hidden_size 128 --batch_size 100 --num_blocks 5 --before_log_epoch 1000 --NoDecoder --select 2

python test.py  --autoencoder LSA --estimator MAF --dataset mnist --score_normed  --hidden_size 128 --num_blocks 5 --batch_size 100 --select 2 --NoDecoder

# python train.py  --NoAutoencoder --estimator MAF --epochs 1000 --dataset mnist --select 2 --lr 0.0001 --hidden_size 1024 --batch_size 100 --num_blocks 5 --before_log_epoch 100 

# python test.py  --NoAutoencoder --estimator MAF --dataset mnist --select 2 --hidden_size 1024 --num_blocks 5 