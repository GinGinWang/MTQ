python test.py --dataset kddcup --autoencoder LSA --lr 0.0001 --Train --batch_size 1024 --epoch 20000 --code_length 1 --select 1
python test.py --dataset kddcup --autoencoder LSA   --lr 0.00001 --Train --batch_size 1024 --epoch 10000 --code_length 1 --select 1 --estimator SOS --load_lsa --Fixed
python test.py --dataset kddcup --autoencoder LSA   --lr 0.00001 --Train --batch_size 1024 --epoch 10000 --code_length 1 --select 1 --estimator SOS --load_lsa --PreTrained
python test.py --dataset kddcup --autoencoder LSA   --lr 0.00001 --Train --batch_size 1024 --epoch 10000 --code_length 1 --select 1 --estimator SOS  --MulObj
