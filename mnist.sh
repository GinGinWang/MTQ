# python test.py --dataset mnist --autoencoder LSA --lr 0.0001   --epoch 1000 --select  8 --estimator SOS --MulObj --Test 

# python test.py --dataset mnist --autoencoder LSA --lr 0.0001   --epoch 1000 --select  8 --estimator SOS --MulObj --Test 
# python test.py --dataset mnist --autoencoder LSA --lr 0.0001   --epoch 1000 --select  8 --estimator SOS --Fixed --load_lsa --Test
# python test.py --dataset mnist --autoencoder LSA --lr 0.0001   --epoch 1000 --select  8 --estimator SOS --PreTrained --load_lsa --Test


# python test.py --dataset mnist --autoencoder LSAD --lr 0.0001   --epoch 1000 --select  8  --Train

# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 2000   --Train --log_step 200 --select 0
# python test.py --dataset mnist --autoencoder LSAD --lr 0.0001   --epoch 2000   --Train --log_step 200 --select 0 


# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 2000   --Train --log_step 200 --select 1
# python test.py --dataset mnist --autoencoder LSAD --lr 0.0001   --epoch 2000   --Train --log_step 200 --select 1

# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 2000   --Train --log_step 200 --select 2
# python test.py --dataset mnist --autoencoder LSAD --lr 0.0001   --epoch 2000   --Train --log_step 200 --select 2

# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 2000   --Train --log_step 200 --select 3
# python test.py --dataset mnist --autoencoder LSAD --lr 0.0001   --epoch 2000   --Train --log_step 200 --select 3

# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 2000   --Train --log_step 200 --select 4
# python test.py --dataset mnist --autoencoder LSAD --lr 0.0001   --epoch 2000   --Train --log_step 200 --select 4

# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 2000   --Train --log_step 200 --select 5
# python test.py --dataset mnist --autoencoder LSAD --lr 0.0001   --epoch 2000   --Train --log_step 200 --select 5

# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 2000   --Train --log_step 200 --select 6
# python test.py --dataset mnist --autoencoder LSAD --lr 0.0001   --epoch 2000   --Train --log_step 200 --select 6

# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 2000   --Train --log_step 200 --select 7
# python test.py --dataset mnist --autoencoder LSAD --lr 0.0001   --epoch 2000   --Train --log_step 200 --select 7

# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 2000   --Train --log_step 200 --select 8
# python test.py --dataset mnist --autoencoder LSAD --lr 0.0001   --epoch 2000   --Train --log_step 200 --select 8

# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 2000   --Train --log_step 200 --select 9
# python test.py --dataset mnist --autoencoder LSAD --lr 0.0001   --epoch 2000   --Train --log_step 200 --select 9

#------------------------------------------------------------------------------------------------------------------

# Train AE
# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 2000   --Test --log_step 200 
# python test.py --dataset mnist --autoencoder LSAD --lr 0.0001   --epoch 2000   --Test --log_step 200
# python test.py --dataset mnist --autoencoder LSA --lr 0.0001   --epoch 2000   --Test --log_step 200

# Train MulObj
python test.py --dataset mnist --autoencoder LSA --lr 0.0001   --epoch 1000 --select  2 --estimator SOS --MulObj --Train
python test.py --dataset mnist --autoencoder LSA --lr 0.0001   --epoch 1000 --select  2 --estimator EN --MulObj --Train
python test.py --dataset mnist --autoencoder LSA --lr 0.0001   --epoch 1000 --select  2 --estimator MAF --MulObj --Train

python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  2 --estimator SOS --MulObj --Train
python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  2 --estimator EN --MulObj --Train
python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  2 --estimator MAF --MulObj --Train

python test.py --dataset mnist --autoencoder LSAD --lr 0.0001   --epoch 1000 --select  2 --estimator SOS --MulObj --Train
python test.py --dataset mnist --autoencoder LSAD --lr 0.0001   --epoch 1000 --select  2 --estimator EN --MulObj --Train
python test.py --dataset mnist --autoencoder LSAD --lr 0.0001   --epoch 1000 --select  2 --estimator MAF --MulObj --Train

python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  8 --estimator SOS --MulObj --Train
python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  8 --estimator EN --MulObj --Train
python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  8 --estimator MAF --MulObj --Train


python test.py --dataset mnist --autoencoder LSAD --lr 0.0001   --epoch 1000 --select  8 --estimator SOS --MulObj --Train
python test.py --dataset mnist --autoencoder LSAD --lr 0.0001   --epoch 1000 --select  8 --estimator EN --MulObj --Train
python test.py --dataset mnist --autoencoder LSAD --lr 0.0001   --epoch 1000 --select  8 --estimator MAF --MulObj --Train


# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  8 --estimator SOS --Fixed --load_lsa --Train
# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  8 --estimator SOS --MulObj --Train
# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  8 --estimator SOS --PreTrained --load_lsa --Train

# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  9 --estimator SOS --Fixed --load_lsa --Train
# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  9 --estimator SOS --MulObj --Train
# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  9 --estimator SOS --PreTrained --load_lsa --Train

# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  7 --estimator SOS --Fixed --load_lsa --Train
# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  7 --estimator SOS --MulObj --Train
# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  7 --estimator SOS --PreTrained --load_lsa --Train

# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  6 --estimator SOS --Fixed --load_lsa --Train
# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  6 --estimator SOS --MulObj --Train
# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  6 --estimator SOS --PreTrained --load_lsa --Train

# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  5 --estimator SOS --Fixed --load_lsa --Train
# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  5 --estimator SOS --MulObj --Train
# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  5 --estimator SOS --PreTrained --load_lsa --Train

# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  4 --estimator SOS --Fixed --load_lsa --Train
# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  4 --estimator SOS --MulObj --Train
# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  4 --estimator SOS --PreTrained --load_lsa --Train

# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  3 --estimator SOS --Fixed --load_lsa --Train
# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  3 --estimator SOS --MulObj --Train
# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  3 --estimator SOS --PreTrained --load_lsa --Train

# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  2 --estimator SOS --Fixed --load_lsa --Train
# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  2 --estimator SOS --MulObj --Train
# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  2 --estimator SOS --PreTrained --load_lsa --Train

# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  1 --estimator SOS --Fixed --load_lsa --Train
# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  1 --estimator SOS --MulObj --Train
# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  1 --estimator SOS --PreTrained --load_lsa --Train

# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  0 --estimator SOS --Fixed --load_lsa --Train
# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  0 --estimator SOS --MulObj --Train
# python test.py --dataset mnist --autoencoder LSAW --lr 0.0001   --epoch 1000 --select  0 --estimator SOS --PreTrained --load_lsa --Train
