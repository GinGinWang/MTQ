# python test.py --autoencoder LSA --dataset fmnist --batch_size 256 --code_length 64 --epochs 300  --lr 0.001 
python test.py  --autoencoder LSA --estimator SOS  --dataset fmnist --batch_size 256 --code_length 64    --epochs 200   --lr 0.0001 --MulObj --num_blocks 1 --hidden_size 2048
