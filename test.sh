python train.py  --autoencoder LSA --estimator MAF --epochs 10000 --dataset mnist  --lr 0.0001 --hidden_size 512 --batch_size 256 --num_blocks 3  --code_length 64 --select 2

python test.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 512 --num_blocks 3 --batch_size 256 --code_length 64 --select 2 --score_normed

# python train.py  --autoencoder LSA --estimator SOS --epochs 10000 --dataset mnist  --lr 0.0001 --hidden_size 128 --batch_size 100 --num_blocks 3 --before_log_epoch 1000 --NoDecoder --select 2

# python test.py  --autoencoder LSA --estimator SOS --dataset mnist   --hidden_size 128 --num_blocks 3 --batch_size 100 --select 2 --NoDecoder 

# python train.py  --NoAutoencoder --estimator MAF --epochs 1000 --dataset mnist --select 2 --lr 0.0001 --hidden_size 1024 --batch_size 100 --num_blocks 5 --before_log_epoch 100 

# python test.py  --NoAutoencoder --estimator MAF --dataset mnist --select 2 --hidden_size 1024 --num_blocks 5 