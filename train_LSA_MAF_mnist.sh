# done
# START_TIME=$(date +%s.%N)

# python train_temp.py  --autoencoder LSA --estimator MAF --epochs 1000 --dataset mnist  --lr 0.1 --hidden_size 1024 --batch_size 256 --num_blocks 10  --code_length 64 --before_log_epochs 100  --PreTrained

# python test_temp.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 1024 --num_blocks 10 --batch_size 256 --code_length 64 --score_normed --PreTrained 

# END_TIME=$(date +%s.%N)
# ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
# echo "Runtime: $ELAPSED_TIME seconds"

# done 
# START_TIME=$(date +%s.%N)

# python train_temp.py  --autoencoder LSA --estimator MAF --epochs 1000 --dataset mnist  --lr 0.01 --hidden_size 1024 --batch_size 256 --num_blocks 10  --code_length 64 --before_log_epochs 100 

# python test_temp.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 1024 --num_blocks 10 --batch_size 256 --code_length 64 --score_normed  

# END_TIME=$(date +%s.%N)
# ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
# echo "Runtime: $ELAPSED_TIME seconds"


python train_temp.py  --autoencoder LSA --estimator MAF --epochs 10000 --dataset mnist  --lr 0.0001 --hidden_size 1024 --batch_size 256 --num_blocks 10  --code_length 64 --select 8 --before_log_epochs 100

python test_temp.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 1024 --num_blocks 10 --batch_size 256 --code_length 64 --score_normed --select 8  
