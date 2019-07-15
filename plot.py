
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = np.load('/home/jj27wang/novelty-detection/NovelDect_SoS/SoSLSA/checkpoints/cifar10_gth/b1h256/0SOS_1_history_test.npz')
# self.test_result_dir = join(checkpoints_dir,f'{cl}{name}_{nameflag}_history_test')

num_bins = 100
print(data['sample_y'].shape)
print(data['sample_llk'].shape)
print(data['sample_rec'].shape)

ind_normal = np.where(data['sample_y'] == 1)
ind_novel = np.where(data['sample_y'] == 0)

string_type = 'llk'
string =  'sample_'+ string_type

label_1 = string_type + '_normal'
label_2 = string_type + '_novel'
be_den =1

maxllk_normal = np.max(data[string][ind_normal])
minllk_normal = np.min(data[string][ind_normal])


maxllk_novel = np.max(data[string][ind_novel])
minllk_novel = np.min(data[string][ind_novel])

print(f"maxllk_normal:{maxllk_normal}")
print(f"minllk_normal:{minllk_normal}")
print(f"maxllk_novel:{maxllk_novel}")
print(f"minllk_novel:{minllk_novel}")

if be_den:
    plt.figure(0)
    sns.distplot(data[string][ind_normal], hist=True, kde=True, 
                 bins=(maxllk_normal-minllk_normal)/5, color = 'darkblue', 
                 hist_kws={'edgecolor':'black'},
                 label = label_1,
                 kde_kws={'linewidth': 4}
                 )
    plt.legend()
    plt.savefig(f'distgraph/gth_cifar_e03_{string}_normal.png')    
    
    plt.figure(1)
    sns.distplot(data[string][ind_novel], hist=True, kde=True, 
                 bins=num_bins, color = 'red', 
                 hist_kws={'edgecolor':'green'},
                 label = label_2,
                 kde_kws={'linewidth': 4})
    plt.legend()
    plt.savefig(f'distgraph/gth_cifar_e03_{string}_novel.png')




plt.figure(2)
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

plt.savefig(f'distgraph/gth_cifar_e03.png')









