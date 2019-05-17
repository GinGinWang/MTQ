# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048 --select 1

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048 --select 2

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048 --select 3

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048 --select 4

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048 --select 5

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048 --select 6

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048 --select 7

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048 --select 8

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048 --select 9



python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --MulObj  --num_blocks 1 --hidden_size 2048 --NoTrain --Qt 

# +-------+----------+-----------+-----------+---------+----------+-----------+----------+
# | Class | AUROC-NS | AUROC-LLK | AUROC-REC |   llk1  |   llk2   | AUROC-NSQ | AUROC-QT |
# +-------+----------+-----------+-----------+---------+----------+-----------+----------+
# |   0   |  0.998   |   0.998   |   0.984   | 169.622 | -66.938  |   0.988   |  0.997   |
# |   1   |  0.999   |   0.999   |   0.999   | 178.615 | -356.648 |   0.999   |  0.998   |
# |   2   |  0.981   |   0.981   |   0.919   | 185.560 | 111.550  |   0.938   |  0.977   |
# |   3   |  0.970   |   0.970   |   0.913   | 190.235 | 114.843  |   0.929   |  0.962   |
# |   4   |  0.977   |   0.977   |   0.921   | 167.685 | -108.874 |   0.942   |  0.980   |
# |   5   |  0.984   |   0.983   |   0.908   | 186.722 |  99.671  |   0.929   |  0.975   |
# |   6   |  0.997   |   0.997   |   0.986   | 187.679 | -396.646 |   0.989   |  0.997   |
# |   7   |  0.981   |   0.981   |   0.962   | 166.441 |  -1.387  |   0.967   |  0.974   |
# |   8   |  0.967   |   0.971   |   0.846   | 193.706 | 109.967  |   0.871   |  0.975   |
# |   9   |  0.990   |   0.991   |   0.951   | 195.583 |  15.149  |   0.962   |  0.992   |
# |  avg  |  0.985   |   0.985   |   0.939   | 182.185 | -47.931  |   0.951   |  0.983   |
# +-------+----------+-----------+-----------+---------+----------+-----------+----------+

