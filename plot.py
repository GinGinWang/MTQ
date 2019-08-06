
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from result_helpers.utils import *

i = 0

np.seterr(divide='ignore', invalid='ignore')

for epoch in [0,50,100,150,200,250,300,350,400]:
    filename = f'0LSA_SOS_mul_history_train_{epoch}'
    data = np.load(f'/home/jj27wang/novelty-detection/NovelDect_SoS/SoSLSA/checkpoints/mnist/b1h2048c32/{filename}.npz')
    test_filename = f'0LSA_SOS_mul_{epoch}_history_test'
    test_data = np.load(f'/home/jj27wang/novelty-detection/NovelDect_SoS/SoSLSA/checkpoints/mnist/b1h2048c32/{test_filename}.npz') 


    num_bins = 100
    print(data['sample_y'].shape)
    print(data['sample_llk'].shape)

    ind_nominal = np.where(data['sample_y'] == 1)
    ind_novel = np.where(data['sample_y'] == 0)

    string_type = 'llk'
    string =  'sample_'+ string_type



    plt.figure(i,figsize=(5,15))
    plt.subplots_adjust(wspace =0.1, hspace =0.1)#调整子图间距
    plt.subplot(411)
    sns.distplot(data['sample_llk'][ind_nominal], hist=True, kde=False, 
                 bins=num_bins, color = 'darkblue', 
                 hist_kws={'edgecolor':'black'},
                 label = "nominal_density",
                 kde_kws={'linewidth': 1}
                 )
    plt.legend()
    plt.subplot(412)
    sns.distplot(np.max(abs(data['sample_u'][ind_nominal]-0.5),axis= 1), hist=True, kde=True, 
                 bins=num_bins, color = 'green', 
                 hist_kws={'edgecolor':'black'},
                 label = "nominal_source_u",
                 kde_kws={'linewidth': 1}
                 )

    plt.legend()

    
    # for test-set
    ind_nominal = np.where(test_data['sample_y'] == 1)
    ind_novel = np.where(test_data['sample_y'] == 0)

    plt.subplot(413)
    sns.distplot(modify_inf(test_data['sample_llk'][ind_nominal]), hist=False, kde=True, color = 'darkblue', 
                 label = 'nominal_density',
                 kde_kws={'linewidth': 1})

    sns.distplot(modify_inf(test_data['sample_llk'][ind_novel]), hist=False, kde=True, 
                  color = 'red', 
                 label = 'novel_density',
                 kde_kws={'linewidth': 1})
    plt.legend()
  
    plt.subplot(414)
    sns.distplot(np.max(abs(test_data['sample_u'][ind_nominal]-0.5),axis =1), hist=True, kde= True,
                 bins=num_bins, color = 'darkblue', 
                 hist_kws={'edgecolor':'black'},
                 label = 'nominal_quantile_away_center_infnorm',
                 kde_kws={'linewidth': 1}
                 )
    
    sns.distplot(np.max(abs(test_data['sample_u'][ind_novel]-0.5),axis=1), hist=True, kde= True,
                 bins=num_bins, color = 'red', 
                 hist_kws={'edgecolor':'black'},
                 label = 'novel_quantile_away_center_infnorm',
                 kde_kws={'linewidth': 1})
    plt.legend()
    plt.savefig(f'distgraph/new_{filename}_d_q_c32.png')
    i = i+1










# final test
plt.figure(i,figsize=(5,5))
plt.subplots_adjust(wspace =0.1, hspace =0.1)#调整子图间距
test_filename = f'0LSA_SOS_mul_history_test'
test_data = np.load(f'/home/jj27wang/novelty-detection/NovelDect_SoS/SoSLSA/checkpoints/mnist/b1h2048c32/{test_filename}.npz') 

# for test-set
ind_nominal = np.where(test_data['sample_y'] == 1)
ind_novel = np.where(test_data['sample_y'] == 0)

plt.subplot(211)
sns.distplot(modify_inf(test_data['sample_llk'][ind_nominal]), hist=False, kde=True, 
             color = 'darkblue', bins=num_bins,hist_kws={'edgecolor':'black'},
             label = 'nominal_density',
             kde_kws={'linewidth': 1})

# plt.legend()

# plt.subplot(212)
sns.distplot(modify_inf(test_data['sample_llk'][ind_novel]), hist=False, kde=True, 
             color = 'red', bins=num_bins,hist_kws={'edgecolor':'black'},
             label = 'novel_density',
             kde_kws={'linewidth': 1})
plt.legend()

plt.subplot(212)
sns.distplot(np.max(abs(test_data['sample_u'][ind_nominal]-0.5),axis=1), hist=True, kde= True,
             bins=num_bins, color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             label = 'nominal_quantile_away_center',
             kde_kws={'linewidth': 1}
             )
# plt.legend()

# plt.subplot(414)
sns.distplot(np.max(abs(test_data['sample_u'][ind_novel]-0.5),axis =1), hist=True, kde= True,
             bins=num_bins, color = 'red', 
             hist_kws={'edgecolor':'black'},
             label = 'novel_quantile_away_center',
             kde_kws={'linewidth': 1}
             )

plt.legend()
plt.savefig(f'distgraph/new_{test_filename}_d_q_c32.png')

