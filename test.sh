python train.py  --autoencoder LSA --estimator MAF --epochs 10000 --dataset mnist  --lr 0.000001 --hidden_size 512 --batch_size 256 --num_blocks 5  --code_length 64  --PreTrained

# python test.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 512 --num_blocks 5 --batch_size 256 --code_length 64 --score_normed

