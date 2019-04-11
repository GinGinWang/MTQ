# train model LSA, LSA_EN, LSA_SOS, LSA_MAF, EN, SOS, MAF on mnist 

echo "Train LSA"
python train.py  --autoencoder LSA --n_class 2 --epochs 10 --dataset mnist


echo "Train LSA_EN"
python train.py  --autoencoder LSA --estimator EN  --n_class 2 --epochs 10 --dataset mnist
python train.py  --autoencoder LSA --estimator EN  --Combine_density --n_class 2 --epochs 10 --dataset mnist

echo "Train LSA_SOS"
python train.py  --autoencoder LSA --estimator SOS  --n_class 2 --epochs 10 --dataset mnist
python train.py  --autoencoder LSA --estimator SOS  --Combine_density --n_class 2 --epochs 10 --dataset mnist

echo "Train LSA_MAF"
python train.py  --autoencoder LSA --estimator MAF  --n_class 2 --epochs 10 --dataset mnist
python train.py  --autoencoder LSA --estimator MAF  --Combine_density --n_class 2 --epochs 10 --dataset mnist

echo "Train EN"
python train.py  --NoAutoencoder --estimator EN --n_class 2 --epochs 10 --dataset mnist

echo "Train SOS" 
python train.py  --NoAutoencoder --estimator SOS --n_class 2 --epochs 10 --dataset mnist


echo "Train MAF"
python train.py  --NoAutoencoder --estimator MAF --n_class 2 --epochs 10  --dataset mnist
