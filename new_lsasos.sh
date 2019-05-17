#LSA_ET_SOS
# 0 1 2 

# python test.py --dataset mnist --autoencoder LSA_ET --estimator SOS  --batch_size 256 --Fixed --PreTrained --premodel LSA  --lr 0.00001 --num_blocks 2 --hidden_size 2048   --NoTrain --select 4 

# Test on whole Framework
# python test.py --dataset mnist --autoencoder LSA --NoTrain

# python test.py --dataset mnist --autoencoder LSA --estimator SOS   --premodel LSA  --lr 0.000001 --num_blocks 2 --hidden_size 2048   --NoTrain --Add 
# #cifar10

# python test.py --dataset cifar10 --autoencoder LSA --NoTrain

python test.py --dataset cifar10 --autoencoder LSA --estimator SOS   --premodel LSA  --num_blocks 1 --hidden_size 4096   --NoTrain --Add  

# LSA_SOS mul 0.995 0.999(1000)
# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 10000  --num_blocks 2 --hidden_size 2048  --MulObj  --NoTrain 

# python test.py  --autoencoder LSA --estimator SOS  --dataset cifar10 --batch_size 256 --code_length 64    --epochs 10000  --num_blocks 1 --hidden_size 4096  --MulObj --select 1 --NoTrain --PreTrained  

# use old results if possible

#LSA_SOS #0,8,2(0.864,5000),3 5 6, 9,1
# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 5000  --num_blocks 2 --hidden_size 2048   --NoTrain 

# retrain 3 for 5000 epochs
# python test.py  --autoencoder LSA --estimator MAF --dataset mnist   --hidden_size 1024 --num_blocks 10 --batch_size 256 --code_length 64  --NoTrain  




