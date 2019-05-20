python test.py --autoencoder LSA --dataset cifar10 --batch_size 256 --code_length 64 --epochs 1000 --lr 0.0001

# python test.py  --autoencoder LSA --estimator SOS  --dataset cifar10 --batch_size 256 --code_length 64    --epochs 30    --lr 0.00001 --num_blocks 1 --hidden_size 2048 --MulObj 

# 0.4281

# python test.py  --autoencoder LSA --estimator SOS  --dataset cifar10 --batch_size 256 --code_length 64    --epochs 1000    --lr 0.00001 --num_blocks 1 --hidden_size 2048 --MulObj  --select 0 --PreTrained  --premodel LSA 
