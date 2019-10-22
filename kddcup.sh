# kddcup.sh


python test.py --dataset kddcup --lr 0.00001  --batch_size 1024 --code_length 2 --epoch 50000 --select  1 --autoencoder LSA --estimator SOS  --Train --log_step 1000  --hidden_size 2048 --MulObj

python test.py --dataset kddcup --lr 0.00001  --batch_size 1024 --code_length 2 --epoch 50000 --select  1 --autoencoder LSA --estimator SOS  --Test --checkpoint b --log_step 1000 --hidden_size 2048 --MulObj