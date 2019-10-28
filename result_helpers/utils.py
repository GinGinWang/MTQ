import numpy as np
#pip install https://github.com/anguswilliams91/CornerPlot/archive/master.zip
# import corner_plot as cp

import numpy as np
import matplotlib.pyplot as plt
import math


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

def compute_density_metric(model_name, sample_ns_t, sample_y):
# Compute precision, recall, f1_score based on threshold
# if we know a/100 is the percentile of novelty samples in testset
 
    # # y = 1 normal, y = 0 novel
    real_nr = float(sum(sample_y==0)/len(sample_y))
    # real_nr = 0.048
    print(f"Real Novelty_Num: {sum(sample_y == 0)} in {len(sample_y)} samples, Novel Ratio= {real_nr}")

    # #based on density(sort first)
    threshold1 = np.percentile(sample_ns_t, real_nr * 100)
    print(f"threshold1:{threshold1}")

    y_hat1 = np.where(sample_ns_t >= threshold1, 1, 0)
    print(f"Density-based, Predicted Novelty_Num: {sum(y_hat1==0)} in {len(y_hat1)} samples")

    precision_den, recall_den, f1_den, _ =  precision_recall_fscore_support((sample_y==0),(y_hat1==0), average= "binary")
    acc_den = accuracy_score((sample_y == 0),(y_hat1 == 0))

    return precision_den, f1_den, recall_den



def compute_quantile_metric(model_name, sample_qinf, sample_y, code_length, quantile_type):
    

    real_nr = float(sum(sample_y==0)/len(sample_y))
    print(f"Real Novelty_Num: {sum(sample_y == 0)} in {len(sample_y)} samples, Novel Ratio= {real_nr}")


    # threshold_q = -pow((1-0.00001),1/code_length)*0.5
    # if quantile_type =='1':
    # 	threshold_q = threshold_q * code_length
    # elif quantile_type == '2':
    # 	threshold_q = threshold_q * math.sqrt(code_length)

    # print(threshold_q) 

    real_threshold = np.percentile(sample_qinf,real_nr * 100)
    print(f"{real_nr}percentage_threshold of quantile:real_threshold, max:{max(sample_qinf)} min:{min(sample_qinf)}")

    y_hat_q = np.where((sample_qinf)>=(real_threshold), 1, 0)
    # y_hat_q = np.where((sample_qinf)>=(threshold_q), 1, 0)

    print(f"Quantile-based, Predicted Novelty_Num: {sum(y_hat_q==0)} in {len(y_hat_q)} samples")
    precision, recall, f1, _ = precision_recall_fscore_support((sample_y==0),(y_hat_q==0), average= "binary")
    return precision, recall, f1


def custom_viz(kernels, path=None, cols=None):
    def set_size(w,h, ax=None):
        
        if not ax: 
            ax=plt.gca()
            l = ax.figure.subplotpars.left
            r = ax.figure.subplotpars.right
            t = ax.figure.subplotpars.top
            b = ax.figure.subplotpars.bottom
            figw = float(w)/(r-l)
            figh = float(h)/(t-b)
            ax.figure.set_size_inches(figw, figh)
        
            N = kernels.shape[0]
            C = kernels.shape[1]

            Tot = N*C

            # If single channel kernel with HxW size,# plot them in a row.# Else, plot image with C number of columns.if C>1:
            columns = C
        elif cols==None:
            columns = N
        elif cols:
            columns = cols
            rows = Tot // columns 
            rows += Tot % columns

            pos = range(1,Tot + 1)

    fig = plt.figure(1)
    fig.tight_layout()
    k=0
    for i in range(kernels.shape[0]):
        for j in range(kernels.shape[1]):
            img = kernels[i][j]
            ax = fig.add_subplot(rows,columns,pos[k])
            ax.imshow(img, cmap='gray')
            plt.axis('off')
            k = k+1

    set_size(30,30,ax)
    
    if path:
        plt.savefig(path, dpi=100)
    
    plt.show()
