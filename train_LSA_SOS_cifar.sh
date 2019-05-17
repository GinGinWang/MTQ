# # # doing 
# START_TIME=$(date +%s.%N)

# python train_temp.py  --autoencoder LSA --estimator SOS --epochs 15 --dataset cifar10  --lr 0.0001 --hidden_size 1024 --batch_size 256 --num_blocks 2  --code_length 64 --before_log_epochs 10  --PreTrained --Fixed

# python test_temp.py  --autoencoder LSA --estimator SOS  --dataset cifar10   --hidden_size 1024 --num_blocks 2 --batch_size 100 --code_length 64  --PreTrained --Fixed  --score_normed

# END_TIME=$(date +%s.%N)
# ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
# echo "Runtime: $ELAPSED_TIME seconds"



# START_TIME=$(date +%s.%N)

# python train_temp.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset cifar10  --lr 0.0001 --hidden_size 1024 --batch_size 256 --num_blocks 2  --code_length 64 --before_log_epochs 10  --PreTrained 

# python test_temp.py  --autoencoder LSA --estimator SOS  --dataset cifar10   --hidden_size 1024 --num_blocks 2 --batch_size 100 --code_length 64  --PreTrained  --score_normed  
# END_TIME=$(date +%s.%N)
# ELAPSED_TIME =$(echo "$END_TIME - $START_TIME" | bc )
# echo "Runtime: $ELAPSED_TIME seconds"





# START_TIME=$(date +%s.%N)

# python train_temp.py  --autoencoder LSA --estimator SOS --epochs 10000 --dataset cifar10  --lr 0.001 --hidden_size 1024 --batch_size 256 --num_blocks 2  --code_length 64 --before_log_epochs 100  

# python test_temp.py  --autoencoder LSA --estimator SOS  --dataset cifar10   --hidden_size 1024 --num_blocks 2 --batch_size 100 --code_length 64  --score_normed 

# # if good on 9, try other numbers on n_class 8

# END_TIME=$(date +%s.%N)
# ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
# echo "Runtime: $ELAPSED_TIME seconds"



# START_TIME=$(date +%s.%N)

# Combine Density

# python train_temp.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset cifar10  --lr 0.001 --hidden_size 1024 --batch_size 256 --num_blocks 2 --code_length 64 --before_log_epochs 200 --Combine_density --select 9 --lam 0.01

# python test_temp.py  --autoencoder LSA --estimator SOS  --dataset cifar10   --hidden_size 1024 --num_blocks 2 --batch_size 100 --code_length 64  --score_normed --Combine_density --select 9 --lam 0.01


# python train_temp.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset cifar10  --lr 0.1 --hidden_size 2048 --batch_size 256 --num_blocks 2 --code_length 64 --before_log_epochs 100 --Combine_density

# python test_temp.py  --autoencoder LSA --estimator SOS  --dataset cifar10   --hidden_size 2048 --num_blocks 2 --batch_size 100 --code_length 64  --score_normed --Combine_density 

# END_TIME=$(date +%s.%N)
# ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
# echo "Runtime: $ELAPSED_TIME seconds"


python test.py  --autoencoder LSA --estimator SOS  --dataset cifar10 --batch_size 256 --code_length 64    --epochs 5000  --num_blocks 1 --hidden_size 4096 --score_normed --MulObj --lr 0.00001 --PreTrained --premodel LSA --select 0

python test.py  --autoencoder LSA --estimator SOS  --dataset cifar10 --batch_size 256 --code_length 64    --epochs 5000  --num_blocks 1 --hidden_size 4096 --score_normed --MulObj --lr 0.00001 --PreTrained --premodel LSA --select 1
python test.py  --autoencoder LSA --estimator SOS  --dataset cifar10 --batch_size 256 --code_length 64    --epochs 5000  --num_blocks 1 --hidden_size 4096 --score_normed --MulObj --lr 0.00001 --PreTrained --premodel LSA --select 2
python test.py  --autoencoder LSA --estimator SOS  --dataset cifar10 --batch_size 256 --code_length 64    --epochs 5000  --num_blocks 1 --hidden_size 4096 --score_normed --MulObj --lr 0.00001 --PreTrained --premodel LSA --select 3
python test.py  --autoencoder LSA --estimator SOS  --dataset cifar10 --batch_size 256 --code_length 64    --epochs 5000  --num_blocks 1 --hidden_size 4096 --score_normed --MulObj --lr 0.00001 --PreTrained --premodel LSA --select 4
python test.py  --autoencoder LSA --estimator SOS  --dataset cifar10 --batch_size 256 --code_length 64    --epochs 5000  --num_blocks 1 --hidden_size 4096 --score_normed --MulObj --lr 0.00001 --PreTrained --premodel LSA --select 5
python test.py  --autoencoder LSA --estimator SOS  --dataset cifar10 --batch_size 256 --code_length 64    --epochs 5000  --num_blocks 1 --hidden_size 4096 --score_normed --MulObj --lr 0.00001 --PreTrained --premodel LSA --select 6
python test.py  --autoencoder LSA --estimator SOS  --dataset cifar10 --batch_size 256 --code_length 64    --epochs 5000  --num_blocks 1 --hidden_size 4096 --score_normed --MulObj --lr 0.00001 --PreTrained --premodel LSA --select 7
python test.py  --autoencoder LSA --estimator SOS  --dataset cifar10 --batch_size 256 --code_length 64    --epochs 5000  --num_blocks 1 --hidden_size 4096 --score_normed --MulObj --lr 0.00001 --PreTrained --premodel LSA --select 8
python test.py  --autoencoder LSA --estimator SOS  --dataset cifar10 --batch_size 256 --code_length 64    --epochs 5000  --num_blocks 1 --hidden_size 4096 --score_normed --MulObj --lr 0.00001 --PreTrained --premodel LSA --select 9