python test.py  --autoencoder LSA --estimator EN  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000  --score_normed  --lr 0.0001 --MulObj --NoTrain 
python test.py  --autoencoder LSA --estimator EN  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000  --score_normed  --lr 0.0001 --MulObj --NoTrain  --checkpoint 200
python test.py  --autoencoder LSA --estimator EN  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000  --score_normed  --lr 0.0001 --MulObj --NoTrain --checkpoint 400
python test.py  --autoencoder LSA --estimator EN  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000  --score_normed  --lr 0.0001 --MulObj --NoTrain  --checkpoint 600
python test.py  --autoencoder LSA --estimator EN  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000  --score_normed  --lr 0.0001 --MulObj --NoTrain  --checkpoint 800

# +-------+----------+-----------+-----------+-----------+-----------+
# | Class | AUROC-NS | AUROC-LLK | AUROC-REC |    llk1   |    llk2   |
# +-------+----------+-----------+-----------+-----------+-----------+
# |   0   |  0.9976  |   0.9982  |   0.9895  | -203.5627 | -564.8151 |
# |   1   |  0.9995  |   0.9984  |   0.9993  | -145.9116 | -537.4307 |
# |   2   |  0.9849  |   0.9783  |   0.9110  | -197.0837 | -309.6766 |
# |   3   |  0.9770  |   0.9707  |   0.9120  | -189.6476 | -310.9517 |
# |   4   |  0.9848  |   0.9850  |   0.9487  | -201.6445 | -393.8183 |
# |   5   |  0.9741  |   0.9613  |   0.9388  | -181.3505 | -240.6457 |
# |   6   |  0.9975  |   0.9960  |   0.9822  | -179.9284 | -466.5758 |
# |   7   |  0.9856  |   0.9787  |   0.9630  | -176.8049 | -314.4034 |
# |   8   |  0.9628  |   0.9753  |   0.8498  | -187.7078 | -307.2203 |
# |   9   |  0.9902  |   0.9876  |   0.9614  | -183.9171 | -352.1704 |
# |  avg  |  0.9854  |   0.9829  |   0.9456  | -184.7559 | -379.7708 |
# +-------+----------+-----------+-----------+-----------+-----------+

python test.py  --autoencoder LSA --estimator EN  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000  --score_normed  --lr 0.0001 --NoTrain 

python test.py  --autoencoder LSA --estimator EN  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000  --score_normed  --lr 0.0001 --NoTrain --checkpoint 200
python test.py  --autoencoder LSA --estimator EN  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000  --score_normed  --lr 0.0001 --NoTrain --checkpoint 400
python test.py  --autoencoder LSA --estimator EN  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000  --score_normed  --lr 0.0001 --NoTrain --checkpoint 600
python test.py  --autoencoder LSA --estimator EN  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000  --score_normed  --lr 0.0001 --NoTrain --checkpoint 800
# +-------+----------+-----------+-----------+-----------+-----------+
# | Class | AUROC-NS | AUROC-LLK | AUROC-REC |    llk1   |    llk2   |
# +-------+----------+-----------+-----------+-----------+-----------+
# |   0   |  0.9972  |   0.9894  |   0.9936  | -215.4801 | -373.4477 |
# |   1   |  0.9990  |   0.9817  |   0.9993  | -164.7928 | -450.3761 |
# |   2   |  0.9792  |   0.9103  |   0.9180  | -183.7441 | -286.2261 |
# |   3   |  0.9736  |   0.9429  |   0.9402  | -177.8414 | -253.3010 |
# |   4   |  0.9765  |   0.9557  |   0.9382  | -186.1636 | -344.3781 |
# |   5   |  0.9772  |   0.9555  |   0.9518  | -181.5981 | -263.1590 |
# |   6   |  0.9978  |   0.9953  |   0.9868  | -179.1381 | -445.1257 |
# |   7   |  0.9834  |   0.9689  |   0.9687  | -180.2647 | -332.7911 |
# |   8   |  0.9757  |   0.9468  |   0.8818  | -192.7726 | -284.8140 |
# |   9   |  0.9892  |   0.9755  |   0.9790  | -177.2934 | -352.2132 |
# |  avg  |  0.9849  |   0.9622  |   0.9557  | -183.9089 | -338.5832 |
# +-------+----------+-----------+-----------+-----------+-----------+
