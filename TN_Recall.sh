# use different novel ratio in testset 
# fix TN/N = 0.9 true_normal/normal = 0.9


# use the full test_set
python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048  --NoTrain --select_novel_classes 3 8 --select 8 

python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048  --novel_ratio 0.8 --NoTrain --select_novel_classes 3 8 --select 8

python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048  --novel_ratio 0.7 --NoTrain --select_novel_classes 3 8 --select 8

python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048  --novel_ratio 0.6 --NoTrain --select_novel_classes 3 8 --select 8

python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048  --novel_ratio 0.5 --NoTrain --select_novel_classes 3 8 --select 8

python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048  --novel_ratio 0.4 --NoTrain --select_novel_classes 3 8 --select 8

python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048  --novel_ratio 0.3 --NoTrain --select_novel_classes 3 8 --select 8 

python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048  --novel_ratio 0.2 --NoTrain --select_novel_classes 3 8 --select 8

python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048  --novel_ratio 0.1 --NoTrain --select_novel_classes 3 8 --select 8

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048  --novel_ratio 0 --NoTrain





python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048  --NoTrain --select_novel_classes 1 8 --select 8 

python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048  --novel_ratio 0.8 --NoTrain --select_novel_classes 1 8 --select 8

python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048  --novel_ratio 0.7 --NoTrain --select_novel_classes 1 8 --select 8

python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048  --novel_ratio 0.6 --NoTrain --select_novel_classes 1 8 --select 8

python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048  --novel_ratio 0.5 --NoTrain --select_novel_classes 1 8 --select 8

python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048  --novel_ratio 0.4 --NoTrain --select_novel_classes 1 8 --select 8

python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048  --novel_ratio 0.3 --NoTrain --select_novel_classes 1 8 --select 8 

python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048  --novel_ratio 0.2 --NoTrain --select_novel_classes 1 8 --select 8

python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048  --novel_ratio 0.1 --NoTrain --select_novel_classes 1 8 --select 8
