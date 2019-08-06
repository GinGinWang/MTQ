
# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --num_blocks 1 --hidden_size 2048 --fixed


# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001 --num_blocks 1 --hidden_size 2048 --fixed --NoTrain 

# +-------+----------+-----------+-----------+----------+----------+------------+----------+--------+--------+
# | Class | AUROC-NS | AUROC-LLK | AUROC-REC | AUROC-q1 | AUROC-q2 | AUROC-qinf | PRCISION |   F1   | RECALL |
# +-------+----------+-----------+-----------+----------+----------+------------+----------+--------+--------+
# |   0   |  0.9946  |   0.9912  |   0.9903  |  0.8328  |  0.9891  |   0.9851   |  0.9894  | 0.9894 | 0.9894 |
# |   1   |  0.9988  |   0.9986  |   0.9992  |  0.9273  |  0.9972  |   0.9965   |  0.9966  | 0.9966 | 0.9966 |
# |   2   |  0.9547  |   0.9431  |   0.9097  |  0.7953  |  0.9157  |   0.9005   |  0.9678  | 0.9678 | 0.9678 |
# |   3   |  0.9615  |   0.9529  |   0.9371  |  0.7730  |  0.9345  |   0.9252   |  0.9725  | 0.9725 | 0.9725 |
# |   4   |  0.9611  |   0.9575  |   0.9349  |  0.9216  |  0.9551  |   0.9500   |  0.9639  | 0.9639 | 0.9639 |
# |   5   |  0.8884  |   0.8286  |   0.9572  |  0.5000  |  0.9243  |   0.9046   |  0.9452  | 0.9452 | 0.9452 |
# |   6   |  0.9918  |   0.9886  |   0.9814  |  0.9102  |  0.9900  |   0.9862   |  0.9871  | 0.9871 | 0.9871 |
# |   7   |  0.9571  |   0.9475  |   0.9639  |  0.7579  |  0.9367  |   0.9271   |  0.9627  | 0.9627 | 0.9627 |
# |   8   |  0.9419  |   0.9381  |   0.8665  |  0.8863  |  0.9327  |   0.9237   |  0.9637  | 0.9637 | 0.9637 |
# |   9   |  0.9842  |   0.9824  |   0.9658  |  0.9172  |  0.9751  |   0.9698   |  0.9835  | 0.9835 | 0.9835 |
# |  avg  |  0.9634  |   0.9528  |   0.9506  |  0.8222  |  0.9550  |   0.9469   |  0.9732  | 0.9732 | 0.9732 |
# +-------+----------+-----------+-----------+----------+----------+------------+----------+--------+--------+



# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1500    --lr 0.0001 --num_blocks 1 --hidden_size 2048 --fixed --select 0

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1500    --lr 0.0001 --num_blocks 1 --hidden_size 2048 --fixed --select 1

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1500    --lr 0.0001 --num_blocks 1 --hidden_size 2048 --fixed --select 2

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1500    --lr 0.0001 --num_blocks 1 --hidden_size 2048 --fixed --select 3

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1500    --lr 0.0001 --num_blocks 1 --hidden_size 2048 --fixed --select 4

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1500    --lr 0.0001 --num_blocks 1 --hidden_size 2048 --fixed --select 5

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1500    --lr 0.0001 --num_blocks 1 --hidden_size 2048 --fixed --select 6

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1500    --lr 0.0001 --num_blocks 1 --hidden_size 2048 --fixed --select 7

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1500    --lr 0.0001 --num_blocks 1 --hidden_size 2048 --fixed --select 8

# python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1500    --lr 0.0001 --num_blocks 1 --hidden_size 2048 --fixed --select 9



python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1500    --lr 0.0001 --num_blocks 1 --hidden_size 2048 --fixed --NoTrain



# +-------+----------+-----------+-----------+----------+----------+------------+----------+--------+--------+
# | Class | AUROC-NS | AUROC-LLK | AUROC-REC | AUROC-q1 | AUROC-q2 | AUROC-qinf | PRCISION |   F1   | RECALL |
# +-------+----------+-----------+-----------+----------+----------+------------+----------+--------+--------+
# |   0   |  0.9935  |   0.9896  |   0.9911  |  0.9606  |  0.9869  |   0.9839   |  0.9867  | 0.9867 | 0.9867 |
# |   1   |  0.9993  |   0.9992  |   0.9994  |  0.9949  |  0.9986  |   0.9984   |  0.9974  | 0.9974 | 0.9974 |
# |   2   |  0.9619  |   0.9578  |   0.9102  |  0.8725  |  0.9416  |   0.9287   |  0.9713  | 0.9713 | 0.9713 |
# |   3   |  0.9637  |   0.9571  |   0.9471  |  0.8963  |  0.9387  |   0.9292   |  0.9725  | 0.9725 | 0.9725 |
# |   4   |  0.9654  |   0.9619  |   0.9390  |  0.9181  |  0.9464  |   0.9396   |  0.9668  | 0.9668 | 0.9668 |
# |   5   |  0.9779  |   0.9693  |   0.9610  |  0.7375  |  0.9505  |   0.9383   |  0.9776  | 0.9776 | 0.9776 |
# |   6   |  0.9893  |   0.9866  |   0.9844  |  0.8172  |  0.9856  |   0.9827   |  0.9852  | 0.9852 | 0.9852 |
# |   7   |  0.9777  |   0.9753  |   0.9575  |  0.9168  |  0.9603  |   0.9532   |  0.9779  | 0.9779 | 0.9779 |
# |   8   |  0.9569  |   0.9563  |   0.8830  |  0.8609  |  0.9366  |   0.9271   |  0.9717  | 0.9717 | 0.9717 |
# |   9   |  0.9800  |   0.9777  |   0.9631  |  0.9389  |  0.9706  |   0.9660   |  0.9815  | 0.9815 | 0.9815 |
# |  avg  |  0.9766  |   0.9731  |   0.9536  |  0.8914  |  0.9616  |   0.9547   |  0.9789  | 0.9789 | 0.9789 |
# +-------+----------+-----------+-----------+----------+----------+------------+----------+--------+--------+


