import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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


def plot_source_dist_by_dimensions(sample, sample_y):

	points_num= len(sample)

	for i in range(points_num):
		point = sample[i,:]
		for j in len(point):
			if sample_y:
				plot.plt(j, point[j,:], 'go',label = label_1)
			else:
				plot.plt(j, point[j,:], 'ro',label = label_2)


	plt.savefig(f'test_source_distribution.png')

