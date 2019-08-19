import numpy as np
#pip install https://github.com/anguswilliams91/CornerPlot/archive/master.zip
# import corner_plot as cp

import numpy as np
import matplotlib.pyplot as plt


import pandas as pd
from pandas.plotting import scatter_matrix



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



	# use cornerplt

	# chains = (sample[normal_idxs, 0:5],sample[novel_idxs, 0:5])

	# cp.multi_corner_plot(chains, axis_labels=columns, linewidth=2.,\
 #                                            chain_labels=["normal","novel"], figsize= (20,20), )

	# plt.savefig(f'distgraph/test.png')
