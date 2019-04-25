
START_TIME=$(date +%s.%N)

python train_temp.py  --autoencoder LSA --estimator MAF --epochs 10000 --dataset mnist  --lr 0.0001 --hidden_size 512 --batch_size 256 --num_blocks 5  --code_length 64 --before_log_epochs 5 --select 9 --lam 0.1

python test_temp.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 512 --num_blocks 5 --batch_size 256 --code_length 64 --score_normed --select 9

python train_temp.py  --autoencoder LSA --estimator MAF --epochs 10000 --dataset mnist  --lr 0.0001 --hidden_size 512 --batch_size 256 --num_blocks 5  --code_length 64 --before_log_epochs 5 -n_class 9 --lam 0.1

python test_temp.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 512 --num_blocks 5 --batch_size 256 --code_length 64 --score_normed --n_class 9

ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
echo "Runtime: $ELAPSED_TIME seconds"

python test_temp.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 512 --num_blocks 5 --batch_size 256 --code_length 64 --score_normed 