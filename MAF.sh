echo "START OUR Expermients!"

echo "MAF MINIST"

# Not combine density#
 echo "Train"
python train.py --estimator MAF --dataset mnist --batch_size 1000  --num_blocks 5 --epochs 10 
