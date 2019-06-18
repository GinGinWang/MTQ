import os
import torch
import numpy as np

from torchvision import datasets

FT_DIR = '/home/jj27wang/novelty-detection/NovelDect_SoS/SoSLSA/baseline_ADT/gtfeatures/'
Data_DIR = '/home/jj27wang/novelty-detection/NovelDect_SoS/SoSLSA/data/CIFAR10_GT/'
name = 'cifar10'

train_dataset = []
test_dataset = []


for idx  in range(10):
	print (idx)
	test_ft_file_name = '{}_transformations_test_{}.npy'.format(name,
	                                         idx)

	test_ft_file_path = os.path.join(FT_DIR, 'cifar10', test_ft_file_name)

	train_ft_file_name = '{}_transformations_train_{}.npy'.format(name,
	                                         idx)

	train_ft_file_path = os.path.join(FT_DIR, 'cifar10', train_ft_file_name)


	# Get train and test split
	train_split = np.load(train_ft_file_path)
	print(len(train_split))
	train_split = np.reshape(train_split, (5000, 72, 32, 32, 3))
	train_split = train_split.transpose(0, 2, 3, 4, 1)
	train_split = np.reshape(train_split, (5000, 32, 32, 216))
	train_dataset = [train_dataset, train_split

	test_split = np.load(test_ft_file_path)
	print(len(test_split))
	test_split = np.reshape(test_split, (1000, 72, 32, 32, 3))
	test_split = test_split.transpose(0, 2, 3, 4, 1)
	test_split = np.reshape(test_split, (1000, 32, 32, 216))
	# test_dataset[idx*1000:(idx+1)*1000-1, :, :,:] = test_split
	test_dataset = [test_dataset, test_split]

	print(len(train_dataset))
	print(len(test_dataset))

	

train_dataset = np.array(train_dataset)
test_dataset = np.array(test_dataset)

print(train_dataset.shape)
print(test_dataset.shape)

np.save(train_dataset, f'{Data_DIR}/cifar10-gt-train' )
np.save(test_dataset, f'{Data_DIR}/cifar10-gt-test')

print('Feature Saved')




