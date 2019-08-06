
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import pandas as pd
from pandas.plotting import scatter_matrix




#data = np.load('../mnist/8LSA_SOS_64_None_0.0001_mul_1_history.npy.npz')
data = np.load('../rings/b20SOS_2_None_0.0001_1_history.npy.npz')
#data = np.load('4AAE_SOS_64_345_1e-06_mul_history.npy.npz')




#plt.plot(np.arange(1,x_end_normal), data['sample_rec'][ind_normal], label = 'rec_normal')
#plt.plot(np.arange(1,x_end_novel), data['sample_rec'][ind_novel], label = 'rec_novel')
#x_end = 200

x_end = len(data['trn_loss'])+1
x_begin = 1
be_process = 0
be_quantile = 1

#print(data['trn_nllk'][0:x_end-1])

if be_process:
    plt.plot(np.arange(x_begin,x_end),data['trn_loss'][x_begin-1:x_end-1],label = 'trn_loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
    
    plt.plot(np.arange(x_begin,x_end),data['trn_rec'][x_begin-1:x_end-1],label = 'trn_rec')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
    
    plt.plot(np.arange(x_begin,x_end),data['trn_nllk'][x_begin-1:x_end-1],label = 'trn_nllk')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


if be_quantile:
    
   print(np.shape(data['sample_uni']))
    
   plt.figure(figsize=(15,20))   
   for i in range(data['sample_uni'].shape[1]):
              
       plt.subplot(11,6,i+1)
       label_i = 'd' + str(i+1)
       sns.distplot(data['sample_uni'][:,i],label = label_i)
       plt.legend(loc='upper right')
   
   plt.show()


  #plt.savefig('train.png')

  #data = pd.DataFrame(data['sample_uni'])   
  #scatter_matrix(data, alpha=0.2, figsize=(6, 6), diagonal='kde')

  #plt.scatter(data['sample_uni'][:, 0], data['sample_uni'][:, 1], color='navy',
  #       marker='.', label='inner')


#data = np.load('1LSA_SOS_mul_history.npy')
#print(data)


#plt.plot(np.arange(1,len(data_val_loss)+1),data_val_loss, label = 'val_loss')
#plt.plot(np.arange(1,len(data_trn_loss)+1),data_trn_loss, label = 'trn_loss')
#plt.plot(np.arange(1,len(data_trn_rec)+1),data_trn_rec, label = 'trn_rec')
#plt.plot(np.arange(1,len(data_trn_nllk)+1),data_trn_nllk, label = 'trn_nllk')












