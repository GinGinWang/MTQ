
python train_temp.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset mnist  --lr 0.1 --hidden_size 2048 --batch_size 256 --num_blocks 2 --code_length 64 --before_log_epochs 100 --Combine_density --n_class 7

python test_temp.py  --autoencoder LSA --estimator SOS  --dataset mnist   --hidden_size 2048 --num_blocks 2 --batch_size 100 --code_length 64  --score_normed --Combine_density --n_class 7
