# python test.py  --dataset cifar10  --estimator SOS  --epoch 10000 --hidden_size 2048 --lr 0.0001  --n_class 6 --NoTrain --checkpoiont b

python test.py  --dataset cifar10_gt --select 1 --autoencoder LSA --estimator SOS  --epoch 10000 --hidden_size 2048 --lr 0.001 --batch_size 256
# python test.py  --dataset cifar10_gt --select 1 --autoencoder LSA --estimator EN  --epoch 10000 --hidden_size 2048 --lr 0.0001 --batch_size 256
