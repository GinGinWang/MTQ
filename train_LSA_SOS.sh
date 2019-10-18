#mnist
python test.py --dataset mnist --autoencoder LSA --lr 0.00001  --Train  --select 2 --estimator SOS --MulObj --num_blocks 1 --log_step 200

python test.py --dataset mnist --autoencoder LSA --lr 0.00001     --Test --checkpoint b  --select 2 --estimator SOS --MulObj --num_blocks 1 --log_step 200

#fashion-mnist
python test.py --dataset fmnist --autoencoder LSA --lr 0.00001  --Train  --select 2 --estimator SOS --MulObj --num_blocks 1 --log_step 200

python test.py --dataset fmnist --autoencoder LSA --lr 0.00001     --Test --checkpoint b  --select 2 --estimator SOS --MulObj --num_blocks 1 --log_step 200


#auto-encoder
python test.py --dataset mnist --autoencoder LSA --lr 0.00001  --Train  --select 2
# two-stage
python test.py --dataset mnist --autoencoder LSA --lr 0.00001  --Train  --select 2 --estimator SOS --Fixed --load_lsa --num_blocks 1 --log_step 200
python test.py --dataset mnist --autoencoder LSA --lr 0.00001  --Test --checkpoint b --select 2 --estimator SOS --Fixed --num_blocks 1 --log_step 200

#warm-start
python test.py --dataset mnist --autoencoder LSA --lr 0.00001  --Train  --select 2 --estimator SOS --PreTrained --load_lsa --num_blocks 1 --log_step 200
python test.py --dataset mnist --autoencoder LSA --lr 0.00001  --Test --checkpoint b  --select 2 --estimator SOS --PreTrained --num_blocks 1 --log_step 200

