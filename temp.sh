# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 10000  --num_blocks 2 --hidden_size 2048 --score_normed --MulObj --select 0 

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 10000  --num_blocks 2 --hidden_size 2048 --score_normed --MulObj --select 1 

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64   --epochs 10000  --num_blocks 2 --hidden_size 2048 --score_normed --MulObj --select 2 

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 10000  --num_blocks 2 --hidden_size 2048 --score_normed --MulObj --select 3 

# stop at 5000
# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 10000  --num_blocks 2 --hidden_size 2048 --score_normed --MulObj --select 4 

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 10000  --num_blocks 2 --hidden_size 2048 --score_normed --MulObj --select 5 

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 10000  --num_blocks 2 --hidden_size 2048 --score_normed --MulObj --select 6 

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 10000  --num_blocks 2 --hidden_size 2048 --score_normed --MulObj --select 7 

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 10000  --num_blocks 2 --hidden_size 2048 --score_normed --MulObj --select 8 

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 10000  --num_blocks 2 --hidden_size 2048 --score_normed --MulObj --select 9 


# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048 --NoTrain --Qt

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048 --NoTrain --Qt  --checkpoint 200

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048 --NoTrain --Qt  --checkpoint 400

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048 --NoTrain --Qt  --checkpoint 600

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048 --NoTrain --Qt  --checkpoint 800


# python test.py  --autoencoder LSA --estimator SOS  --dataset cifar10 --batch_size 256 --code_length 64 --num_blocks 1 --hidden_size 2048  --NoTrain

# python test.py --autoencoder LSA --dataset fmnist --batch_size 256 --code_length 64 --checkpoint b  --lr 0.001 --NoTrain 

# python test.py  --autoencoder LSA --estimator EN  --dataset fmnist --batch_size 256 --code_length 64    --score_normed  --MulObj --NoTrain  --checkpoint b


# python test.py  --autoencoder LSA --estimator SOS  --dataset fmnist --batch_size 256 --code_length 64  --MulObj --num_blocks 1 --hidden_size 2048  --NoTrain  --Qt --checkpoint b 

# python test.py  --autoencoder LSA --estimator EN  --dataset fmnist --batch_size 256 --code_length 64    --epochs 1000  --score_normed  --lr 0.001 --NoTrain 

# python test.py  --autoencoder LSA --estimator SOS  --dataset cifar10 --batch_size 256 --code_length 64   --num_blocks 1 --hidden_size 2048 --MulObj --select 0 --PreTrained --premodel LSA  --NoTrain --checkpoint 400

# python test.py --autoencoder LSA --dataset cifar10 --code_length 64 --NoTrain --n_class 9


# python test.py  --autoencoder LSA --estimator SOS  --dataset cifar10 --batch_size 256 --code_length 64    --num_blocks 1 --hidden_size 2048 --NoTrain --Qt  

# python test.py  --autoencoder LSA --estimator SOS  --dataset cifar10 --batch_size 256 --code_length 64    --num_blocks 1 --hidden_size 2048 --select 0 --NoTrain --Qt --checkpoint b --MulObj --PreTrained --premodel LSA

# python test.py  --NoAutoencoder  --estimator SOS  --dataset thyroid --num_blocks 1 --hidden_size 2048  --select 0 --lr 0.0001  --checkpoint b --NoTrain

# python test.py  --NoAutoencoder  --estimator SOS  --dataset thyroid --batch_size 100 --num_blocks 1 --hidden_size 1024 --lr 0.00001 --epochs 10000 --NoTrain --select 0 --Qt


# python test.py  --autoencoder LSA --estimator SOS  --dataset cifar10 --batch_size 100 --code_length 64    --epochs 200    --lr 0.0000001 --num_blocks 1 --hidden_size 2048 --MulObj --NoTrain --checkpoint b



# python test.py  --autoencoder LSA --estimator SOS  --dataset cifar10 --batch_size 256 --code_length 64    --epochs 200    --lr 0.001  --num_blocks 1 --hidden_size 2048 --lam 0.01 --NoTrain --checkpoint b

python test.py  --autoencoder LSA --estimator SOS  --dataset cifar10 --batch_size 256 --code_length 64    --epochs 200    --lr 0.001  --num_blocks 1 --hidden_size 2048  --lam 0.0001 --NoTrain  --Qt --n_class 5