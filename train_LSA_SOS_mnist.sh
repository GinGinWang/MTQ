# doing 
# START_TIME=$(date +%s.%N)

python train_temp.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset mnist  --lr 0.1 --hidden_size 2048 --batch_size 256 --num_blocks 1  --code_length 64 --before_log_epochs 10  --PreTrained --Fixed --select 9

python test_temp.py  --autoencoder LSA --estimator SOS  --dataset mnist   --hidden_size 2048 --num_blocks 1 --batch_size 256 --code_length 64 --score_normed --PreTrained --Fixed --select 9

# END_TIME=$(date +%s.%N)
# ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
# echo "Runtime: $ELAPSED_TIME seconds"

# python train_temp.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset mnist  --lr 0.001 --hidden_size 1024 --batch_size 256 --num_blocks 10  --code_length 64 --before_log_epochs 10  --PreTrained --Fixed --select 2

