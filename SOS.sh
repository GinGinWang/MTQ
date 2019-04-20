python train.py  --autoencoder LSA --estimator SOS --epochs 10000 --dataset mnist  --lr 0.01 --hidden_size 2048 --batch_size 256 --num_blocks 5  --code_length 64  --PreTrained --select 2

python test.py  --autoencoder LSA --estimator SOS --dataset mnist   --hidden_size 2048 --num_blocks 5 --batch_size 256 --code_length 64 --score_normed --select 2