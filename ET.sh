# python test.py --dataset mnist --autoencoder LSA_ET --estimator SOS  --batch_size 256 --Fixed --PreTrained --premodel LSA   --num_blocks 2 --hidden_size 2048 --epochs 4001 --lr 0.000001

# try 5000 epochs for better results
python test.py --dataset cifar10 --autoencoder LSA_ET --estimator SOS  --batch_size 256 --Fixed --PreTrained --premodel LSA  --lr 0.0001 --num_blocks 1 --hidden_size 4096 --epochs 5000



