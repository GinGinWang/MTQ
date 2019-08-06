import numpy as np 
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# for epoch in [100,200, 300,400,500,600,700,800,900]:
for epoch in [0,50,100,150,200,250,300,350,400]:
	filename = f'0LSA_SOS_mul_history_train_{epoch}'
	data = np.load(f'/home/jj27wang/novelty-detection/NovelDect_SoS/SoSLSA/checkpoints/mnist/b1h2048c64/{filename}.npz')
	sample = data['sample_u']
	sample_y = data['sample_y']

	points_num, dimension = sample.shape




	idxs = range(points_num)
	normal_idxs = [idx for idx in idxs if sample_y[idx] == 1]

	novel_idxs =  [idx for idx in idxs if sample_y[idx] == 0]
	# print(normal_idxs)
	# print(novel_idxs)
	print("Normal_num:", len(normal_idxs))
	print("Novel_num:", len(novel_idxs))

	columns = ['d'+str(i) for i in range(dimension) ]


	# # scatter_matrix
	#first make some fake data with same layout as yours
	# data_normal = pd.DataFrame(sample[normal_idxs, :], columns=columns)
	# data_novel = pd.DataFrame(sample[novel_idxs, :], columns = columns)
	# # now plot using pandas

	# scatter_matrix(data_normal, alpha=0.2, figsize=(50, 50), diagonal='hist', color = 'b',marker='o',hist_kwds={'bins':20})
	# plt.savefig(f'distgraph/{filename}.png')

	# if (len(novel_idxs)>0):
	# 	# for test-set
	# 	scatter_matrix(data_novel, alpha=0.2, figsize=(50, 50), diagonal='hist', color = 'r', marker='o',hist_kwds={'bins':20})
	# 	plt.savefig(f'distgraph/{filename}_novel.png')


	be_quantile =1 

	if be_quantile:
	    
	   print(np.shape(data['sample_u']))
	    
	   plt.figure(figsize=(15,20))   
	   for i in range(data['sample_u'].shape[1]):
	              
	       plt.subplot(11,6,i+1)
	       plt.xlim(0,1)
	       label_i = 'd' + str(i+1)
	       sns.distplot(data['sample_u'][:,i],label = label_i,kde=True)
	       plt.legend(loc='upper right')
	   
	   plt.show()

	plt.savefig(f'distgraph/train_{epoch}.png')
	plt.close()
