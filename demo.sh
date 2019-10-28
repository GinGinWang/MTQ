# Train LSA-SOS on mnist, setting classs 1 as the normal class
## Using MGD optimization
python main.py --dataset mnist  --lr 0.00001  --select 1 --autoencoder LSA --estimator SOS  --Train  --epochs 1000 --MulObj
python main.py --dataset mnist  --lr 0.00001  --select 1 --autoencoder LSA --estimator SOS  --Test  --MulObj


# Train LSA-SOS on fmnist, setting classs 1 as the normal class
## Using MGD optimization
python main.py --dataset fmnist  --lr 0.00001  --select 1 --autoencoder LSA --estimator SOS  --Train  --epochs 1000 --MulObj
python main.py --dataset fmnist  --lr 0.00001  --select 1 --autoencoder LSA --estimator SOS  --Test  --MulObj




# Experiment on kddcup
python main.py --dataset kddcup --lr 0.00001  --batch_size 1024 --code_length 2 --epochs 50000 --select  1 --autoencoder LSA --estimator SOS  --Train --log_step 1000  --hidden_size 2048 --MulObj

python main.py --dataset kddcup --lr 0.00001  --batch_size 1024 --code_length 2 --epochs 50000 --select  1 --autoencoder LSA --estimator SOS  --Test --checkpoint b --log_step 1000 --hidden_size 2048 --MulObj

# +-------+----------+-----------+-----------+----------+----------+------------+----------+--------+--------+--------------+--------+-----------+--------------+--------+-----------+----------------+---------+-------------+
# | Class | AUROC-NS | AUROC-LLK | AUROC-REC | AUROC-q1 | AUROC-q2 | AUROC-qinf | PRCISION |   F1   | RECALL | precision_q1 | f1_q1  | recall_q1 | precision_q2 | f1_q2  | recall_q2 | precision_qinf | f1_qinf | recall_qinf |
# +-------+----------+-----------+-----------+----------+----------+------------+----------+--------+--------+--------------+--------+-----------+--------------+--------+-----------+----------------+---------+-------------+
# |   0   |  0.7885  |   0.9696  |   0.7885  |  0.9716  |  0.9694  |   0.9693   |  0.9622  | 0.9622 | 0.9622 |    0.9621    | 0.9621 |   0.9621  |    0.9622    | 0.9622 |   0.9622  |     0.9622     |  0.9622 |    0.9622   |
# |  avg  |  0.7885  |   0.9696  |   0.7885  |  0.9716  |  0.9694  |   0.9693   |  0.9622  | 0.9622 | 0.9622 |    0.9621    | 0.9621 |   0.9621  |    0.9622    | 0.9622 |   0.9622  |     0.9622     |  0.9622 |    0.9622   |
# +-------+----------+-----------+-----------+----------+----------+------------+----------+--------+--------+--------------+--------+-----------+--------------+--------+-----------+----------------+---------+-------------+

# Experiment on thyroid
python main.py --dataset thyroid --lr 0.001  --batch_size 1024 --code_length 6 --epochs 50000 --select  1 --estimator SOS  --Train --log_step 2000 --hidden_size 2048
python main.py --dataset thyroid --lr 0.001  --batch_size 1024 --code_length 6 --epochs 50000 --select  1 --estimator SOS  --Test --checkpoint b --log_step 2000 --hidden_size 2048

# +-------+--------+----------+--------+--------+--------------+--------+-----------+--------------+--------+-----------+----------------+---------+-------------+
# | Class | AUROC  | PRCISION |   F1   | RECALL | precision_q1 | f1_q1  | recall_q1 | precision_q2 | f1_q2  | recall_q2 | precision_qinf | f1_qinf | recall_qinf |
# +-------+--------+----------+--------+--------+--------------+--------+-----------+--------------+--------+-----------+----------------+---------+-------------+
# |   0   | 0.9869 |  0.7312  | 0.7312 | 0.7312 |    0.5269    | 0.5269 |   0.5269  |    0.5806    | 0.5806 |   0.5806  |     0.7527     |  0.7527 |    0.7527   |
# |  avg  | 0.9869 |  0.7312  | 0.7312 | 0.7312 |    0.5269    | 0.5269 |   0.5269  |    0.5806    | 0.5806 |   0.5806  |     0.7527     |  0.7527 |    0.7527   |
# +-------+--------+----------+--------+--------+--------------+--------+-----------+--------------+--------+-----------+----------------+---------+-------------+


