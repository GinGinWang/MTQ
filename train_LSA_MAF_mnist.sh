# done
# START_TIME=$(date +%s.%N)

# python train_temp.py  --autoencoder LSA --estimator MAF --epochs 1000 --dataset mnist  idden_size 1024 --batch_size 256 --num_blocks 10  --code_length 64 --PreTrained

# python test_temp.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 1024 --num_blocks 10 --batch_size 256 --code_length 64 --score_normed --PreTrained 

# END_TIME=$(date +%s.%N)
# ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
# echo "Runtime: $ELAPSED_TIME seconds"

# done 
# START_TIME=$(date +%s.%N)

# python train_temp.py  --autoencoder LSA --estimator MAF --epochs 1000 --dataset mnist  hidden_size 1024 --batch_size 256 --num_blocks 10  --code_length 64 
# python test_temp.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 1024 --num_blocks 10 --batch_size 256 --code_length 64 --score_normed  

# END_TIME=$(date +%s.%N)
# ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
# echo "Runtime: $ELAPSED_TIME seconds"



# python test.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 1024 --num_blocks 10 --batch_size 256 --code_length 64 --score_normed --epochs 1000  --select 0

#doing
python test.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 1024 --num_blocks 10 --batch_size 256 --code_length 64 --score_normed --epochs 5000   --select 0 --MulObj
python test.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 1024 --num_blocks 10 --batch_size 256 --code_length 64 --score_normed --epochs 5000   --select 1 --MulObj
python test.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 1024 --num_blocks 10 --batch_size 256 --code_length 64 --score_normed --epochs 5000   --select 2 --MulObj
# retrain
python test.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 1024 --num_blocks 10 --batch_size 256 --code_length 64 --score_normed --epochs 5000   --select 3 --MulObj
python test.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 1024 --num_blocks 10 --batch_size 256 --code_length 64 --score_normed --epochs 5000   --select 4 --MulObj
python test.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 1024 --num_blocks 10 --batch_size 256 --code_length 64 --score_normed --epochs 5000   --select 5 --MulObj
python test.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 1024 --num_blocks 10 --batch_size 256 --code_length 64 --score_normed --epochs 5000   --select 6 --MulObj
python test.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 1024 --num_blocks 10 --batch_size 256 --code_length 64 --score_normed --epochs 5000   --select 7 --MulObj
python test.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 1024 --num_blocks 10 --batch_size 256 --code_length 64 --score_normed --epochs 5000   --select 8 --MulObj
python test.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 1024 --num_blocks 10 --batch_size 256 --code_length 64 --score_normed --epochs 5000   --select 9 --MulObj

# python test.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 1024 --num_blocks 10 --batch_size 256 --code_length 64 --score_normed --epochs 5000   --NoTrain --MulObj