echo "Test score_normed combine_density"
# novel from 0.1 to 0.5
# for i in 0.1 0.2 0.3 0.4 0.5 1
for i in 0.1
	do 
		echo "Test on novel_ratio=$i" 
		python test.py --autoencoder LSA --estimator MAF --dataset cifar10 --num_blocks 10 --combine_density True --score_normed True  --novel_ratio $i
	done