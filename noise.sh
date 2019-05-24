# noise.sh
python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 5000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048  --select 8 --noise 0.01


python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048  --select 8 --noise 0.02


python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048  --select 8 --noise 0.03


python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048  --select 8 --noise 0.04


python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048  --select 8 --noise 0.05
