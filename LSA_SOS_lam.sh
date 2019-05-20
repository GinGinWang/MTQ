# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001   --num_blocks 1 --hidden_size 2048  --lam 1 --NoTrain --Qt
# +-------+----------+-----------+-----------+---------+---------+-----------+----------+
# | Class | AUROC-NS | AUROC-LLK | AUROC-REC |   llk1  |   llk2  | AUROC-NSQ | AUROC-QT |
# +-------+----------+-----------+-----------+---------+---------+-----------+----------+
# |   0   |  0.970   |   0.775   |   0.965   | 453.613 | 449.399 |   0.966   |  0.753   |
# |   1   |  0.731   |   0.604   |   0.986   | 208.931 | 128.214 |   0.986   |  0.626   |
# |   2   |  0.800   |   0.748   |   0.790   | 529.392 | 528.298 |   0.792   |  0.747   |
# |   3   |  0.861   |   0.671   |   0.853   | 517.446 | 509.972 |   0.854   |  0.649   |
# |   4   |  0.865   |   0.566   |   0.880   | 515.021 | 513.198 |   0.879   |  0.537   |
# |   5   |  0.643   |   0.465   |   0.732   | 499.131 | 499.366 |   0.730   |  0.448   |
# |   6   |  0.753   |   0.523   |   0.875   | 404.534 | 345.098 |   0.875   |  0.463   |
# |   7   |  0.915   |   0.764   |   0.911   | 492.171 | 468.252 |   0.911   |  0.693   |
# |   8   |  0.798   |   0.540   |   0.786   | 517.035 | 516.486 |   0.787   |  0.532   |
# |   9   |  0.645   |   0.420   |   0.879   | 442.251 | 446.644 |   0.878   |  0.402   |
# |  avg  |  0.798   |   0.607   |   0.866   | 457.952 | 440.493 |   0.866   |  0.585   |
# +-------+----------+-----------+-----------+---------+---------+-----------+----------+
python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001   --num_blocks 1 --hidden_size 2048  --lam 0.1 --NoTrain --Qt
# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001   --num_blocks 1 --hidden_size 2048  --lam 0.01 --NoTrain --Qt
# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001   --num_blocks 1 --hidden_size 2048  --lam 10 --NoTrain --Qt






