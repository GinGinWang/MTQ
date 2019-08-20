import numpy as np
#pip install https://github.com/anguswilliams91/CornerPlot/archive/master.zip
# import corner_plot as cp

import numpy as np
import matplotlib.pyplot as plt


import pandas as pd
from pandas.plotting import scatter_matrix

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def modify_inf(result):
	# change nan to zero
	result[result!=result]=0

	# change inf to 10**20
	rsize = len(result)
	
	for i in range(0,rsize):
		
		if np.isinf(result[i]):
			if result[i]>0:
				result[i] = +10**20
			else:
				result[i] = -10**20
	return result


def plot_source_dist_by_dimensions(sample, sample_y, filename):
	# latent vector # m-dimension
	# plot graphs for pairwise joint distribution/ marginal distribution
	# 

	points_num, dimension = sample.shape

	idxs = range(points_num)
	normal_idxs = [idx for idx in idxs if sample_y[idx] == 1]
	
	novel_idxs =  [idx for idx in idxs if sample_y[idx] == 0]
	# print(normal_idxs)
	# print(novel_idxs)
	print("Normal_num:", len(normal_idxs))
	print("Novel_num:", len(novel_idxs))

	columns = ['z'+str(i) for i in range(dimension) ]
	

	# scatter_matrix
	#first make some fake data with same layout as yours
	data_normal = pd.DataFrame(sample[normal_idxs, :], columns=columns)
	data_novel = pd.DataFrame(sample[novel_idxs, :], columns = columns)
	# now plot using pandas

	scatter_matrix(data_normal, alpha=0.2, figsize=(50, 50), diagonal='hist', color = 'b',marker='o',hist_kwds={'bins':20})
	plt.savefig(f'{filename}_normal.png')

	scatter_matrix(data_novel, alpha=0.2, figsize=(50, 50), diagonal='hist', color = 'r', marker='o',hist_kwds={'bins':20})
	plt.savefig(f'{filename}_novel.png')

def compute_metric(model_name, sample_ns_t, sample_y):
# Compute precision, recall, f1_score based on threshold
# if we know a/100 is the percentile of novelty samples in testset
 
    # # y = 1 normal, y = 0 novel
    real_nr= float(sum(sample_y==0)/len(sample_y))            
    print(f"Real Novelty_Num: {sum(sample_y == 0)} in {len(sample_y)} samples, Novel Ratio= {real_nr}")


    # #based on density(sort first)
    threshold1 = np.percentile(sample_ns_t, real_nr*100)
    print(f"threshold1:{threshold1}")

    y_hat1 = np.where(sample_ns_t >= threshold1, 1, 0)
    print(f"Density-based, Predicted Novelty_Num: {sum(y_hat1==0)} in {len(y_hat1)} samples")
    wrong_predict1 = np.where(sample_y!= y_hat1)
    print(f"Wrongly Predict on {len(wrong_predict1)}")                

    precision_den, recall_den, f1_den, _ =  precision_recall_fscore_support((sample_y==0),(y_hat1==0), average= "binary")
    acc_den = accuracy_score((sample_y==0),(y_hat1==0))

    return precision_den, recall_den,f1_den