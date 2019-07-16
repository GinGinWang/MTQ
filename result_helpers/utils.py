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

	print(sample.shape)
	print(sample_y.shape)
	points_num= len(sample)

	idxs = range(points_num)
	normal_idxs = [idx for idx in idxs if sample_y[idx] == 1]
	
	novel_idxs =  [idx for idx in idxs if sample_y[idx] == 0]
	# print(normal_idxs)
	# print(novel_idxs)
	print("Normal_num:", len(normal_idxs))
	print("Novel_num:", len(novel_idxs))

	# normal points
	x = range(64)
	plt.figure(0)
	# plt.savefig(f'distgraph/test_normal_distribution.png')

	# plt.figure(1)
	aver_normal = []
	aver_novel = []
	aver_diff = []
	for i in range (64):
		aver_normal.append(sum([sample[idx, i] for idx in normal_idxs]))
		aver_novel.append( sum([sample[idx, i] for idx in novel_idxs]))
		aver_diff.append(abs(aver_normal[i]-aver_novel[i]))

    
	print(aver_diff)

	idxmax1 , idxmax2 = sorted(range(len(aver_diff)), key=lambda i: aver_diff[i])[-2:]

	print(idxmax1)
	print(idxmax2)

	plt.plot(sample[novel_idxs, idxmax1], sample[novel_idxs, idxmax2], 'r+')
	
	plt.plot(sample[normal_idxs, idxmax1], sample[normal_idxs,idxmax2], 'b.')

	plt.show()

	plt.savefig(f'distgraph/test_source_distribution.png')



# def plot_1d_dist(sample, sample_y):
 
# def plot_2d_dist(sample, sample_y):

# def plot_3d_dist(sample, sample_y):

