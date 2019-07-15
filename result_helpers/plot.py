
import numpy as np
import matplotlib.pyplot as plt




#data = np.load('../mnist/1LSA_SOS_64_mul_history.npy_test.npz')
data = np.load('/home/jj27wang/novelty-detection/NovelDect_SoS/SoSLSA/checkpoints/cifar10_gth/combinedFalse/PtrFalse/b1h256/9SOS_mul_history.npy_test.npz')
num_bins = 100

ind_normal = np.where(data['sample_y'] == 1)
ind_novel = np.where(data['sample_y'] == 0)

#print(data['sample_rec'][ind_normal])
string_type = 'llk'
string =  'sample_'+ string_type
label_1 = string_type + '_normal'
label_2 = string_type + '_novel'

x_end_normal = len(data[string][ind_normal])+1
x_end_novel = len(data[string][ind_novel])+1

be_CDF = 1
be_sep = 0

a = np.mean(data[string][ind_normal])
b = np.mean(data[string][ind_novel])
print('normal mean = ', a, 'novel mean = ', b)

c = np.count_nonzero(data[string][ind_normal]==0)
d = np.count_nonzero(data[string][ind_novel]==0)
print('number of zeros for normal: ', c, 'number of zeros for novel: ',d)

########CDF plot##############
if be_CDF:
    counts, bin_edges = np.histogram (data[string][ind_normal], bins=num_bins)
    cdf = np.cumsum (counts)
    plt.plot (bin_edges[1:], cdf/cdf[-1],label = label_1)
    counts, bin_edges = np.histogram (data[string][ind_novel], bins=num_bins)
    cdf = np.cumsum (counts)
    plt.plot (bin_edges[1:], cdf/cdf[-1],label = label_2)
    plt.xlim(-5000,5000)
        
    plt.legend()
    plt.show()

else:
    if be_sep:
        plt.hist(data[string][ind_normal], bins=num_bins, label = label_1, color = 'green')
        plt.legend()
        plt.show()
        
        plt.hist(data[string][ind_novel], bins=num_bins, label = label_2, color = 'red')
        #plt.xlim(-5000,5000)
        plt.legend()
        plt.show()
    
    else:
    #    fig, axs = plt.subplots(1, 2, sharey=True)
    #    axs[0].hist(data[string][ind_normal], bins=num_bins, label = label_1)
    #    axs[1].hist(data[string][ind_novel], bins=num_bins, label = label_2)
        plt.hist(data[string][ind_normal], bins=num_bins, label = label_1, alpha=0.5, color = 'green')    
        plt.hist(data[string][ind_novel], bins=num_bins, label = label_2, alpha = 0.5,color = 'red')
        
        plt.xlim(-5000,5000)
        plt.legend()
        plt.show()


plt.plot(data[string][ind_normal],'go',label = label_1)
#plt.ylim(-5000,5000)
#plt.plot(np.mean(data[string][ind_normal])*np.ones(shape = (len(data[string][ind_normal]))),color ='green')        
plt.legend()
plt.show()

plt.plot(data[string][ind_novel],'ro',label = label_2)
#plt.ylim(-5000,5000)
#plt.plot(np.mean(data[string][ind_novel])*np.ones(shape = (len(data[string][ind_novel]))),color = 'red')
plt.legend()
plt.show()











