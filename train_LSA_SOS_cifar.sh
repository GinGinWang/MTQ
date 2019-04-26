# # doing 
START_TIME=$(date +%s.%N)

python train_temp.py  --autoencoder LSA --estimator SOS --epochs 15 --dataset cifar10  --lr 0.0001 --hidden_size 1024 --batch_size 256 --num_blocks 2  --code_length 64 --before_log_epochs 10  --PreTrained --Fixed

python test_temp.py  --autoencoder LSA --estimator SOS  --dataset cifar10   --hidden_size 1024 --num_blocks 2 --batch_size 100 --code_length 64  --PreTrained --Fixed  --score_normed

END_TIME=$(date +%s.%N)
ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
echo "Runtime: $ELAPSED_TIME seconds"



START_TIME=$(date +%s.%N)

python train_temp.py  --autoencoder LSA --estimator SOS --epochs 1000 --dataset cifar10  --lr 0.0001 --hidden_size 1024 --batch_size 256 --num_blocks 2  --code_length 64 --before_log_epochs 10  --PreTrained 

python test_temp.py  --autoencoder LSA --estimator SOS  --dataset cifar10   --hidden_size 1024 --num_blocks 2 --batch_size 100 --code_length 64  --PreTrained  --score_normed  
END_TIME=$(date +%s.%N)
ELAPSED_TIME =$(echo "$END_TIME - $START_TIME" | bc )
echo "Runtime: $ELAPSED_TIME seconds"





START_TIME=$(date +%s.%N)

python train_temp.py  --autoencoder LSA --estimator SOS --epochs 10000 --dataset cifar10  --lr 0.001 --hidden_size 1024 --batch_size 256 --num_blocks 2  --code_length 64 --before_log_epochs 100  

python test_temp.py  --autoencoder LSA --estimator SOS  --dataset cifar10   --hidden_size 1024 --num_blocks 2 --batch_size 100 --code_length 64  --score_normed 

# if good on 9, try other numbers on n_class 8

END_TIME=$(date +%s.%N)
ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc )
echo "Runtime: $ELAPSED_TIME seconds"