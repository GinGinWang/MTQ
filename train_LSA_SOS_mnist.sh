
# START_TIME=$(date +%s.%N)

# python train_temp.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset mnist  --lr 0.0001 --hidden_size 2048 --batch_size 256 --num_blocks 2  --code_length 64--PreTrained --Fixed

# python test_temp.py  --autoencoder LSA --estimator SOS  --dataset mnist   --hidden_size 2048 --num_blocks 2 --batch_size 100 --code_length 64   --score_normed  --PreTrained --Fixed

# python train_temp.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset mnist  --lr 0.0001 --hidden_size 2048 --batch_size 256 --num_blocks 2  --code_length 64--PreTrained

# python test_temp.py  --autoencoder LSA --estimator SOS  --dataset mnist   --hidden_size 2048 --num_blocks 2 --batch_size 100 --code_length 64   --score_normed  --PreTrained 

# END_TIME=$(date +%s.%N)
# ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
# echo "Runtime: $ELAPSED_TIME seconds"


# START_TIME=$(date +%s.%N)

# python train_temp.py  --autoencoder LSA --estimator SOS --epochs 5000 --dataset mnist  --lr 0.0001 --hidden_size 2048 --batch_size 256 --num_blocks 2 --code_length 64

# python test_temp.py  --autoencoder LSA --estimator SOS  --dataset mnist   --hidden_size 2048 --num_blocks 2 --batch_size 100 --code_length 64  --score_normed

# END_TIME=$(date +%s.%N)
# ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
# echo "Runtime: $ELAPSED_TIME seconds"


# test whether combine density works
# START_TIME=$(date +%s.%N)

# python train_temp.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset mnist  --lr 0.0001 --hidden_size 2048 --batch_size 256 --num_blocks 2 --code_length 64--Combine_density

# python test_temp.py  --autoencoder LSA --estimator SOS  --dataset mnist   --hidden_size 2048 --num_blocks 2 --batch_size 100 --code_length 64  --score_normed --Combine_density 

# END_TIME=$(date +%s.%N)
# ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
# echo "Runtime: $ELAPSED_TIME seconds"

# doing now



# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 5000  --num_blocks 2 --hidden_size 2048 --score_normed  --select 0

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 5000  --num_blocks 2 --hidden_size 2048 --score_normed  --select 8

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 5000  --num_blocks 2 --hidden_size 2048 --score_normed  --select 2

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 5000  --num_blocks 2 --hidden_size 2048 --score_normed  --select 3

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 5000  --num_blocks 2 --hidden_size 2048 --score_normed  --select 5

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 5000  --num_blocks 2 --hidden_size 2048 --score_normed  --select 6

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 5000  --num_blocks 2 --hidden_size 2048 --score_normed  --select 9

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 5000  --num_blocks 2 --hidden_size 2048 --score_normed  --select 1

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 5000  --num_blocks 2 --hidden_size 2048 --score_normed  --select 7

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 5000  --num_blocks 2 --hidden_size 2048 --score_normed  --select 4

python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 5000  --num_blocks 2 --hidden_size 2048 --NoTrain --score_normed