#doing
# START_TIME=$(date +%s.%N)

# python train_temp.py  --autoencoder LSA --estimator MAF --epochs 1000 --dataset cifar10  --lr 0.001 --hidden_size 1024 --batch_size 256 --num_blocks 10  --code_length 64 --before_log_epochs 5  --PreTrained --Fixed 

# python test_temp.py  --autoencoder LSA --estimator MAF --dataset cifar10   --hidden_size 1024 --num_blocks 10 --batch_size 256 --code_length 64 --score_normed --PreTrained --Fixed

# END_TIME=$(date +%s.%N)
# ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
# echo "Runtime: $ELAPSED_TIME seconds"

# START_TIME=$(date +%s.%N)

# python train_temp.py  --autoencoder LSA --estimator MAF --epochs 1000 --dataset cifar10  --lr 0.0001 --hidden_size 1024 --batch_size 256 --num_blocks 10  --code_length 64 --before_log_epochs 5  --PreTrained

# python test_temp.py  --autoencoder LSA --estimator MAF --dataset cifar10   --hidden_size 1024 --num_blocks 10 --batch_size 256 --code_length 64 --score_normed --PreTrained 

# END_TIME=$(date +%s.%N)
# ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
# echo "Runtime: $ELAPSED_TIME seconds"

START_TIME=$(date +%s.%N)

python train_temp.py  --autoencoder LSA --estimator MAF --epochs 1000 --dataset cifar10  --lr 0.01 --hidden_size 1024 --batch_size 256 --num_blocks 10  --code_length 64 --before_log_epochs 5 

python test_temp.py  --autoencoder LSA --estimator MAF --dataset cifar10   --hidden_size 1024 --num_blocks 10 --batch_size 256 --code_length 64 --score_normed  

END_TIME=$(date +%s.%N)
ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
echo "Runtime: $ELAPSED_TIME seconds"
