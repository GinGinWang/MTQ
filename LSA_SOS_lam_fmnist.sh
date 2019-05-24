# LSA_SOS_lam_fmnist.sh
python test.py  --autoencoder LSA --estimator SOS  --dataset fmnist --batch_size 256 --code_length 64    --epochs 1000   --lr 0.001  --num_blocks 1 --hidden_size 2048 --lam 1 
python test.py  --autoencoder LSA --estimator SOS  --dataset fmnist --batch_size 256 --code_length 64    --epochs 1000   --lr 0.001  --num_blocks 1 --hidden_size 2048 --lam 0.1
python test.py  --autoencoder LSA --estimator SOS  --dataset fmnist --batch_size 256 --code_length 64    --epochs 1000   --lr 0.001  --num_blocks 1 --hidden_size 2048 --lam 0.01
python test.py  --autoencoder LSA --estimator SOS  --dataset fmnist --batch_size 256 --code_length 64    --epochs 1000   --lr 0.001  --num_blocks 1 --hidden_size 2048 --lam 10

