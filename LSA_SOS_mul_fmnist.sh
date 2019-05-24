# done 0.９10
# python test.py --autoencoder LSA --dataset fmnist --batch_size 256 --code_length 64 --epochs 300  --lr 0.001
# Trying  
# best at 300,350
# python test.py  --autoencoder LSA --estimator SOS  --dataset fmnist --batch_size 256 --code_length 64    --epochs 2000   --lr 0.00001 --MulObj --num_blocks 1 --hidden_size 2048 --select 0
# 
python test.py  --autoencoder LSA --estimator SOS  --dataset fmnist --batch_size 256 --code_length 64    --epochs 1000   --lr 0.00001 --MulObj --num_blocks 1 --hidden_size 2048 --NoTrain --Qt