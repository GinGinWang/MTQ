
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



#data = np.load('../mnist/8SOS_64_1_history.npy_test.npz')
data = np.load('/home/jj27wang/novelty-detection/NovelDect_SoS/SoSLSA/checkpoints/cifar10_gth/b1h256/0SOS_1_history_test.npz')

num_bins = 10

ind_normal = np.where(data['sample_y'] == 1)
ind_novel = np.where(data['sample_y'] == 0)

#print(data['sample_rec'][ind_normal])
string_type = 'llk'
string =  'sample_'+ string_type
label_1 = string_type + '_normal'
label_2 = string_type + '_novel'

x_end_normal = len(data[string][ind_normal])+1
x_end_novel = len(data[string][ind_novel])+1

be_CDF = 0
be_hist = 1
be_den = 0

be_sep = 1

a = np.mean(data[string][ind_normal])
b = np.mean(data[string][ind_novel])
print('normal mean = ', a, 'novel mean = ', b)

c = np.count_nonzero(data[string][ind_normal]==0)
d = np.count_nonzero(data[string][ind_novel]==0)
print('number of zeros for normal: ', c, 'number of zeros for novel: ',d)


####density###########
if be_den:
    sns.distplot(data[string][ind_normal], hist=True, kde=True, 
                 bins=num_bins, color = 'darkblue', 
                 hist_kws={'edgecolor':'black'},
                 label = label_1,
                 kde_kws={'linewidth': 4})
#    sns.distplot(data[string][ind_normal],
#             label = label_1)
        
    plt.legend()
    plt.show()
    
    sns.distplot(data[string][ind_novel], hist=True, kde=True, 
                 bins=num_bins, color = 'darkblue', 
                 hist_kws={'edgecolor':'black'},
                 label = label_2,
                 kde_kws={'linewidth': 4})
    plt.legend()
    plt.show()


if be_CDF:
    counts, bin_edges = np.histogram (data[string][ind_normal], bins=num_bins)
    cdf = np.cumsum (counts)
    plt.plot (bin_edges[1:], cdf/cdf[-1],label = label_1)
    counts, bin_edges = np.histogram (data[string][ind_novel], bins=num_bins)
    cdf = np.cumsum (counts)
    plt.plot (bin_edges[1:], cdf/cdf[-1],label = label_2)
    #plt.xlim(-5000,5000)
        
    plt.legend()
    plt.show()



if be_hist:
    if be_sep:
        plt.figure(0)
        plt.hist(data[string][ind_normal], bins=num_bins, label = label_1, color = 'green')
        plt.legend()
        plt.show()
        plt.savefig(f'distgraph/gth_cifar_normal.png')
        

        plt.figure(1)
        plt.hist(data[string][ind_novel], bins=num_bins, label = label_2, color = 'red')
        #plt.xlim(-5000,5000)
        plt.legend()
        plt.show()
        plt.savefig(f'distgraph/gth_cifar_novel.png')

    
    else:
    #    fig, axs = plt.subplots(1, 2, sharey=True)
    #    axs[0].hist(data[string][ind_normal], bins=num_bins, label = label_1)
    #    axs[1].hist(data[string][ind_novel], bins=num_bins, label = label_2)
        plt.hist(data[string][ind_normal], bins=num_bins, label = label_1, alpha=0.5, color = 'green')    
        plt.hist(data[string][ind_novel], bins=num_bins, label = label_2, alpha = 0.5,color = 'red')
        
        #plt.xlim(-5000,5000)
        plt.legend()
        plt.show()


# plt.plot(data[string][ind_normal],'go',label = label_1)
# plt.ylim(min(data[string][ind_normal]),max(data[string][ind_normal]))
# plt.plot(np.mean(data[string][ind_normal])*np.ones(shape = (len(data[string][ind_normal]))),color ='green')        
# plt.legend()
# plt.show()
# #
# plt.plot(data[string][ind_novel],'ro',label = label_2)
# plt.ylim(min(data[string][ind_novel]),max(data[string][ind_novel]))
# plt.plot(np.mean(data[string][ind_novel])*np.ones(shape = (len(data[string][ind_novel]))),color = 'red')
# plt.legend()
# plt.show()



#print(data[string][ind_normal])
#print(data[string][ind_novel])

print('normal min:', min(data[string][ind_normal]))
print('normal max:', max(data[string][ind_normal]))
print('novel min:', min(data[string][ind_novel]))
print('novel max:', max(data[string][ind_novel]))

if min(data[string][ind_normal]) > max(data[string][ind_novel]):
    print('No overlap!')
else:
    print('Overlap!')








