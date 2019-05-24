# UCI_test.sh
python test.py  --autoencoder LSA  --estimator SOS  --dataset kddcup --batch_size 1024 --num_blocks 1 --hidden_size 2048  --select 0 --lr 0.00001 --epochs 200 --code_length 1 --MulObj --NoTrain --Qt

#SOS
# python test.py  --NoAutoencoder  --estimator SOS  --dataset thyroid --batch_size 100 --num_blocks 1 --hidden_size 1024  --select 0 --lr 0.00001 --epochs 10000 --NoTrain --Qt


