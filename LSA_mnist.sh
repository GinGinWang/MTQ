
python train.py  --autoencoder LSA  --epochs 1000 --dataset mnist

python test.py  --autoencoder LSA  --epochs 1000 --dataset mnist --score_normed

python test.py  --autoencoder LSA  --epochs 1000 --dataset mnist 


python train.py  --autoencoder LSA  --epochs 1000 --dataset mnist --Combine_density 

python test.py  --autoencoder LSA  --epochs 1000 --dataset mnist --Combine_density --score_normed

python test.py  --autoencoder LSA  --epochs 1000 --dataset mnist --Combine_density 


##################################################
python train.py  --autoencoder LSA  --epochs 1000 --dataset cifar10 

python test.py  --autoencoder LSA  --epochs 1000 --dataset cifar10 --score_normed

python test.py  --autoencoder LSA  --epochs 1000 --dataset cifar10 


python train.py  --autoencoder LSA  --epochs 1000 --dataset cifar10 --Combine_density 

python test.py  --autoencoder LSA  --epochs 1000 --dataset cifar10 --Combine_density --score_normed

python test.py  --autoencoder LSA  --epochs 1000 --dataset cifar10 --Combine_density 
