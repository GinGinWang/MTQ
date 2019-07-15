# python test.py  --dataset cifar10  --estimator SOS  --epoch 10000 --hidden_size 2048 --lr 0.0001  --NoTrain --checkpoiont b

# python test.py  --dataset cifar10_gth --estimator SOS  --epoch 10000 --hidden_size 512 --lr 0.0001 --batch_size 64 --select 0 --NoTrain --checkpoint b

# +-------+----------+----------+-------+--------+---------------+-------------------+----------+----------+------------+
# | Class | AUROC-NS | PRCISION |   F1  | RECALL |      llk1     |        llk2       | AUROC-q1 | AUROC-q2 | AUROC-qinf |
# +-------+----------+----------+-------+--------+---------------+-------------------+----------+----------+------------+
# |   0   |  0.999   |  0.999   | 0.999 | 0.999  | -23313673.742 | -389480710682.915 |  0.790   |  1.000   |   1.000    |
# |  avg  |  0.999   |  0.999   | 0.999 | 0.999  | -23313673.742 | -389480710682.915 |  0.790   |  1.000   |   1.000    |
# +-------+----------+----------+-------+--------+---------------+-------------------+----------+----------+------------+

# hidden_size 256 --batch-size 32
# +-------+----------+----------+-------+--------+---------------+-------------------+----------+----------+------------+
# | Class | AUROC-NS | PRCISION |   F1  | RECALL |      llk1     |        llk2       | AUROC-q1 | AUROC-q2 | AUROC-qinf |
# +-------+----------+----------+-------+--------+---------------+-------------------+----------+----------+------------+
# |   0   |  0.999   |  0.999   | 0.999 | 0.999  | -16535124.093 | -134048046333.816 |  0.791   |  1.000   |   1.000    |
# |  avg  |  0.999   |  0.999   | 0.999 | 0.999  | -16535124.093 | -134048046333.816 |  0.791   |  1.000   |   1.000    |
# +-------+----------+----------+-------+--------+---------------+-------------------+----------+----------+------------+


# python test.py  --dataset cifar10_gth --estimator SOS  --epoch 10000 --hidden_size 256 --lr 0.0001 --batch_size 32 --select 0

# python test.py  --dataset cifar10_gt --select 1 --autoencoder LSA --estimator SOS  --epoch 10000 --hidden_size 2048 --lr 0.01 --batch_size 256

python test.py  --dataset cifar10_gth --estimator SOS  --epochs 10000 --hidden_size 256 --lr 0.0001 --batch_size 32 --select 0 --NoTrain --checkpoint b

# python test.py  --dataset fmnist_gth --estimator SOS  --epochs 10000 --hidden_size 256 --lr 0.0001 --batch_size 32 --select 0 --NoTrain --checkpoint b
