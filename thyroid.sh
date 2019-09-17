python test.py --dataset thyroid --autoencoder LSA --lr 0.0001 --batch_size 1 --code_length 1 --epoch 20000 --select  1 --Test

# python test.py --dataset thyroid --autoencoder LSA --lr 0.00001  --batch_size 2 --code_length 2 --epoch 100000 --select  1 --estimator SOS --MulObj --Test

# python test.py --dataset thyroid --autoencoder LSA --lr 0.00001  --batch_size 1024 --code_length 1 --epoch 50000 --select  1 --estimator SOS --Fixed --load_lsa --Test

# python test.py --dataset thyroid --autoencoder LSA --lr 0.00001  --batch_size 1024 --code_length 1 --epoch 50000 --select  1 --estimator SOS --PreTrained --load_lsa --Test




# python test.py --dataset thyroid --autoencoder LSA --lr 0.00001  --batch_size 1024 --code_length 1 --epoch 50000 --select  1 --estimator SOS --MulObj --compute_AUROC 

# python test.py --dataset thyroid --autoencoder LSA --lr 0.00001  --batch_size 1024 --code_length 1 --epoch 50000 --select  1 --estimator SOS --Fixed --load_lsa --compute_AUROC

# python test.py --dataset thyroid --autoencoder LSA --lr 0.00001  --batch_size 1024 --code_length 1 --epoch 50000 --select  1 --estimator SOS --PreTrained --load_lsa --compute_AUROC




# python test.py --dataset thyroid --autoencoder LSA --lr 0.00001  --batch_size 1024 --code_length 1 --epoch 50000 --select  1 --estimator SOS --MulObj --plot_training_loss_auroc 

# python test.py --dataset thyroid --autoencoder LSA --lr 0.00001  --batch_size 1024 --code_length 1 --epoch 50000 --select  1 --estimator SOS --Fixed --load_lsa --plot_training_loss_auroc

# python test.py --dataset thyroid --autoencoder LSA --lr 0.00001  --batch_size 1024 --code_length 1 --epoch 50000 --select  1 --estimator SOS --PreTrained --load_lsa --plot_training_loss_auroc