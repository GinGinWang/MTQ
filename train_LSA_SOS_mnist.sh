# doing 
# START_TIME=$(date +%s.%N)

# python train_temp.py  --autoencoder LSA --estimator SOS --epochs 15 --dataset mnist  --lr 0.0001 --hidden_size 1024 --batch_size 256 --num_blocks 2  --code_length 64 --before_log_epochs 10  --PreTrained --Fixed

# python test_temp.py  --autoencoder LSA --estimator SOS  --dataset mnist   --hidden_size 1024 --num_blocks 2 --batch_size 100 --code_length 64  --PreTrained --Fixed  --score_normed 

# END_TIME=$(date +%s.%N)
# ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
# echo "Runtime: $ELAPSED_TIME seconds"

# START_TIME=$(date +%s.%N)

# python train_temp.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset mnist  --lr 0.0001 --hidden_size 1024 --batch_size 256 --num_blocks 2  --code_length 64 --before_log_epochs 10  --PreTrained 

# python test_temp.py  --autoencoder LSA --estimator SOS  --dataset mnist   --hidden_size 1024 --num_blocks 2 --batch_size 100 --code_length 64  --PreTrained  --score_normed 
# END_TIME=$(date +%s.%N)
# ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
# echo "Runtime: $ELAPSED_TIME seconds"

START_TIME=$(date +%s.%N)

python train_temp.py  --autoencoder LSA --estimator SOS --epochs 10000 --dataset mnist  --lr 0.0001 --hidden_size 2048 --batch_size 256 --num_blocks 2 --code_length 64 --before_log_epochs 10   --select 2 

# python train_temp.py  --autoencoder LSA --estimator SOS --epochs 10000 --dataset mnist  --lr 0.0001 --hidden_size 2048 --batch_size 256 --num_blocks 2 --code_length 64 --before_log_epochs 10   --select 4 
# 
# python train_temp.py  --autoencoder LSA --estimator SOS --epochs 10000 --dataset mnist  --lr 0.0001 --hidden_size 2048 --batch_size 256 --num_blocks 2 --code_length 64 --before_log_epochs 10   --select 5 

# python train_temp.py  --autoencoder LSA --estimator SOS --epochs 10000 --dataset mnist  --lr 0.0001 --hidden_size 2048 --batch_size 256 --num_blocks 2 --code_length 64 --before_log_epochs 10   --select 6 

# python train_temp.py  --autoencoder LSA --estimator SOS --epochs 10000 --dataset mnist  --lr 0.0001 --hidden_size 2048 --batch_size 256 --num_blocks 2 --code_length 64 --before_log_epochs 10   --select 8 

# python train_temp.py  --autoencoder LSA --estimator SOS --epochs 10000 --dataset mnist  --lr 0.0001 --hidden_size 2048 --batch_size 256 --num_blocks 2 --code_length 64 --before_log_epochs 10   --select 9


python test_temp.py  --autoencoder LSA --estimator SOS  --dataset mnist   --hidden_size 2048 --num_blocks 2 --batch_size 100 --code_length 64  --score_normed

END_TIME=$(date +%s.%N)
ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
echo "Runtime: $ELAPSED_TIME seconds"