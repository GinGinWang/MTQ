import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from result_helpers.utils import *
import pickle

model_name = "LSA_EN"
# train_strategy = "mul"
dataset_name = 'fmnist'
class_name_list = ['2','3','4','5','6','8'] # mnist
# class_name_list = ['2']
# xmin = 0
# xmax = 3000
# ymin = 0.9
# ymax = 1.0

for class_name in class_name_list:
	file_name = f'{dataset_name}_c{class_name}_{model_name}_mul_auroc'

	pickle_in = open(file_name,"rb")
	auroc_mul = pickle.load(pickle_in)

	file_name = f'{dataset_name}_c{class_name}_{model_name}_fix_auroc'

	pickle_in = open(file_name,"rb")
	auroc_fix = pickle.load(pickle_in)

	# file_name = f'{dataset_name}_c{class_name}_{model_name}_prt_auroc'

	# pickle_in = open(file_name,"rb")
	# auroc_prt = pickle.load(pickle_in)

	
	# print(auroc.keys())
	auroc_len = len(auroc_mul['ns'])

	x = range(0, auroc_len*100, 100)


	fig = plt.figure(0)
	# plt.axis([xmin, xmax, ymin, ymax])

	plt.plot(x, auroc_mul['nllk'], 'bo-',label = 'mul_nllk_auroc')
	plt.plot(x, auroc_fix['nllk'], 'g--',label = 'fix_nllk_auroc')
	# plt.plot(x, auroc_prt['nllk'], 'r>-',label = 'prt_nllk_auroc')
	# plt.plot(x, auroc['q1'], 'co-',label = 'q1')
	# plt.plot(x, auroc['q2'], 'm--',label = 'q2')
	# plt.plot(x, auroc['qinf'], 'y<-',label = 'qinf')

	plt.legend(loc='lower right')
	# Method 1
	plt.savefig(f'distgraph/compare_{dataset_name}_{model_name}_{class_name}.png')
	plt.close(0)
