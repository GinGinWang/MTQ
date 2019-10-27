import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns

def violin_graph(cl, model_style, epoch_list, metric_list, using_train_set= True):

    for  metric in metric_list:
        frames = []
        for epoch in epoch_list:
            if using_train_set:
                if epoch ==1000:
                    data = np.load(f'checkpoints/mnist/b1h2048c64/{cl}{model_style}_test_train_sample.npz')
                else:
                    data = np.load(f'checkpoints/mnist/b1h2048c64/{cl}{model_style}_{epoch}_test_train_sample.npz')

            else:
                if epoch == 1000:
                    data = np.load(f'checkpoints/mnist/b1h2048c64/{cl}{model_style}_test_sample.npz')
                else:
                    data = np.load(f'checkpoints/mnist/b1h2048c64/{cl}{model_style}_{epoch}_test_sample.npz')


            print(data.files)

            data_dict = {col: data[col].tolist() for col in data.files}
            data_dict['label'] = [ 'nominal' if labeli == 1 else'novel' for labeli in data_dict['label'] ]
            df = pd.DataFrame.from_dict(data_dict)
            frame = df[['label', metric]].copy()
            # frame = dfllk.rename(columns={"llk": "value"})
            frame["epoch"] = [str(epoch)]*len(frame['label'])
            frames.append(frame)

        matplotlib.rc('xtick', labelsize=20) 
        matplotlib.rc('ytick', labelsize=20) 
        matplotlib.rc('font', weight='bold',size=20)
        matplotlib.rc ('axes', labelsize = 20)



        df_final = pd.concat(frames)

        plt.plot()
        if using_train_set:
            sns.violinplot(x="epoch", order=['0', '20', '40', '60', '80', '100'], y=metric, data= df_final, scale="width", split= False, size = 2, aspect =2 ) # epoch 0, 200,1000
            plt.tight_layout()

            plt.savefig(f"violin_graphs/{cl}{model_style}_{metric}_vl_trainset.png")
            plt.close()

        else:
            sns.violinplot(x="epoch", order=['0', '20', '40', '60', '80', '100'], y=metric, data= df_final, hue = "label", scale="width", split= True, size = 2, aspect =2 ) # epoch 0, 200,1000
            plt.tight_layout()

            plt.savefig(f"violin_graphs/{cl}{model_style}_{metric}_vl_testset.png")
            plt.close()


def pairwise_plots(cl, model_style, epoch, select_dimensions, split_flag, using_train_set, graph_type):

    # for distribution u

        frames = []
        if using_train_set:
                if epoch ==1000:
                    data = np.load(f'checkpoints/mnist/b1h2048c64/{cl}{model_style}_test_train_sample.npz')
                else:
                    data = np.load(f'checkpoints/mnist/b1h2048c64/{cl}{model_style}_{epoch}_test_train_sample.npz')

        else:
                if epoch == 1000:
                    data = np.load(f'checkpoints/mnist/b1h2048c64/{cl}{model_style}_test_sample.npz')
                else:
                    data = np.load(f'checkpoints/mnist/b1h2048c64/{cl}{model_style}_{epoch}_test_sample.npz')

        print(data.files)

        data_dict = {col: data[col].tolist() for col in data.files}
        data_dict['label'] = ['nominal' if labeli==1 else'novel' for labeli in data_dict['label'] ]
        for i in range(64):
            data_dict[f'u{i}'] = [item[i]for item in data_dict['u']]





        df = pd.DataFrame.from_dict(data_dict)
        if using_train_set == False:
            df_novel= df.loc[df['label'] == 'novel']
            df_nominal = df.loc[df['label'] == 'nominal']


        # print(df.head)

        if graph_type == "single-2d":
            # for i in range(len(select_dimensions)):
            #     for j in range(len(select_dimensions)):
            #         if i < j:
            

            
            fig = plt.plot()
            # plt.legend(loc=2, prop={'size': 20})

            if using_train_set:
                df_select = df[select_dimensions].copy()
                sns.set_context("paper", rc={"font.size":10,"axes.titlesize":10,"axes.labelsize":10, "axes.lengendsize":10})   

                g = sns.pairplot(df_select, size=1.5, aspect=1)
                plt.savefig(f"u2d_plots/{cl}{model_style}_u0123_{epoch}_trainset.png")
                

            else:
                matplotlib.rc ('axes', labelsize = 20)

                df_select = df[['u56','u57','u58','label']].copy()
                sns.set_context("paper", rc={"font.size":10,"axes.titlesize":10,"axes.labelsize":10, "axes.lengendsize":10})   

                g = sns.pairplot(df_select, hue= 'label', size=1.5, aspect=1)
                plt.savefig(f"u2d_plots/{cl}{model_style}_u0123_{epoch}_testset.png")

            
            plt.close()





        elif graph_type == "marginals":
            # matplotlib.rc('xtick', labelsize=4)
            # matplotlib.rc('ytick', labelsize=4)
            k = 0

            if using_train_set:

                fig, axes = plt.subplots(8, 8, sharex=True, sharey= True, figsize=(10,10))
                for i in range(8):
                    for j in range(8):
                       plt.xlim(0, 1)
                       plt.ylim(0, 5)

                       sns.distplot(df[f"u{k}"], ax=axes[i,j] )
                       k = k + 1

                plt.savefig(f"u2d_plots/{cl}{model_style}_{epoch}_trainset.png")
                # plt.tight_layout()
                plt.close()
            else:
                fig, axes = plt.subplots(8, 8, sharex=True, sharey=True, figsize=(10,10))
                for i in range(8):
                    for j in range(8):
                       plt.xlim(0, 1)
                       plt.ylim(0, 5)
                       l1 = sns.distplot(df_novel[f"u{k}"], ax=axes[i,j],color = 'red' )
                       l2 = sns.distplot(df_nominal[f"u{k}"], ax=axes[i,j],color = 'blue' )
                       k = k+1

                fig.legend([l1, l2],     # The line objects
                   labels=['u_novel','u_nominal'],   # The labels for each line
                   loc=2,   # Position of legend
                   borderaxespad=0.001,    # Small spacing around legend box  # Title for the legend
                   )
                plt.savefig(f"u2d_plots/{cl}{model_style}_{epoch}_testset.png")
                plt.close()


        # pairwise-plot





if __name__ == '__main__':

    model_style = 'LSA_SOS_mul'
    epoch_list = [1000] # multiple epochs 0,100,200,1000
    metric_list = ['NLL','TQM1','TQM2','TQM_inf']
    select_dimensions = ['u56','u57','u58']
    split_flag = True
    using_train_set = True
    graph_type = "single-2d"

    
    for cl in [1]:
        if graph_type == "violin":
            violin_graph(cl, model_style, epoch_list, metric_list, using_train_set)
        else:
            for epoch in epoch_list:
                print("Generating graphs for epoch{i}")
                pairwise_plots(cl, model_style, epoch, select_dimensions, split_flag, using_train_set,  graph_type)
