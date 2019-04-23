START_TIME=$(date +%s.%N)

python train_temp.py  --autoencoder LSA --estimator SOS --epochs 10000 --dataset cifar10  --lr 0.0001 --hidden_size 512 --batch_size 256 --num_blocks 5  --code_length 64 --before_log_epochs 5 --select 0 --lam 0.1

python test_temp.py  --autoencoder LSA --estimator MAF --dataset cifar10   --hidden_size 512 --num_blocks 5 --batch_size 256 --code_length 64 --score_normed --select 0

ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
echo "Runtime: $ELAPSED_TIME seconds"

