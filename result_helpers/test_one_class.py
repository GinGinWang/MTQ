from os.path import join
from typing import Tuple

import numpy as np
import torch
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.base import OneClassDataset
from models.base import BaseModule
from models.loss_functions import SumLoss
from datasets.utils import novelty_score
from datasets.utils import normalize


from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# from result_helpers import metric_method as mm



class OneClassTestHelper(object):
    """
    Performs tests for one-class datasets (MNIST or CIFAR-10).
    """

    def __init__(self, dataset, model, score_normed, novel_ratio, checkpoints_dir, output_file):
        # type: (OneClassDataset, BaseModule, str, str) -> None
        """
        Class constructor.

        :param dataset: dataset class.
        :param model: py-torch model to evaluate.
        :param score_normed: 1 normalized the novelty score with valid set, 0: not normalized
        :param novel_ratio: novel_ratio in test sets
        :param checkpoints_dir: directory holding checkpoints for the model.
        :param output_file: text file where to save results.
        
        """
        self.dataset = dataset
        self.model = model
        self.checkpoints_dir = checkpoints_dir
        self.output_file = output_file

        # control novel ratio in test sets.
        self.novel_ratio = novel_ratio

        # normalized novelty score
        self.score_normed = score_normed
        # Set up loss function
        self.loss = SumLoss(self.model.name)

    @torch.no_grad()
    #temporarily set all the requires_graph flag to false
    def test_one_class_classification(self):
        # type: () -> None
        """
        Actually performs tests.
        """

        # Prepare a table to show results
        oc_table = self.empty_table

        # Set up container for metrics from all classes
        all_metrics = []

        # Start iteration over classes
        for cl_idx, cl in enumerate(self.dataset.test_classes):

            # Load the checkpoint
            self.model.load_w(join(self.checkpoints_dir, f'{cl}{self.model.name}.pkl'))

            if self.score_normed == True:
                # we need a run on validation, to compute
                # normalizing coefficient of the Novelty Score (Eq.9 in LSA)
                min_llk, max_llk, min_rec, max_rec = self.compute_normalizing_coefficients(cl)

            # Prepare test set for one class and control the novel ratio in test set
            dataset = self.dataset
            
            dataset.test(cl,self.novel_ratio)
            loader = DataLoader(self.dataset)

            sample_llk = np.zeros(shape=(len(loader),))
            sample_rec = np.zeros(shape=(len(loader),))
            sample_y = np.zeros(shape=(len(loader),))
<<<<<<< HEAD
            # density = 0
=======
<<<<<<< e68b04d9643bf8aa75b53953df98d650ab4d948c
            density = 0
=======
            # density = 0
>>>>>>> message
>>>>>>> temp

            for i, (x, y) in tqdm(enumerate(loader), desc=f'Computing scores for {dataset}'):
                x = x.to('cuda')
                x_r, z, z_dist, s, log_jacob_s = self.model(x)
                self.loss(x, x_r, z, z_dist,s,log_jacob_s)

                sample_llk[i] = - self.loss.autoregression_loss
                sample_rec[i] = - self.loss.reconstruction_loss
                sample_y[i] = y.item()
                # print (sample_llk[i])
<<<<<<< HEAD
                # density += sample_llk[i]
            
            # density = density/dataset.length
=======
<<<<<<< e68b04d9643bf8aa75b53953df98d650ab4d948c
                density += sample_llk[i]
            
            density = density/dataset.length
=======
                # density += sample_llk[i]
            
            # density = density/dataset.length
>>>>>>> message
>>>>>>> temp

            if self.score_normed == True:
                print(f'min_llk:{min_llk},max_llk:{max_llk}'
                    )
                print(f'min_rec:{min_rec},max_llk:{max_rec}')

                # Normalize scores
                sample_llk = normalize(sample_llk, min_llk, max_llk)

                sample_rec = normalize(sample_rec, min_rec, max_rec)

            #print(sample_llk)
            # Compute the normalized novelty score
<<<<<<< HEAD
=======
<<<<<<< e68b04d9643bf8aa75b53953df98d650ab4d948c
            sample_ns = novelty_score(sample_llk, sample_rec)

            # Compute precision, recall, f1_score based on threshold
            threshold = self.compute_threshold(cl)
            y_hat = np.less(sample_ns, threshold)

            precision = precision_score(sample_y,y_hat)
            f1 = f1_score(sample_y, y_hat)
            recall = recall_score(sample_y, y_hat)
=======
>>>>>>> temp
            print (sample_llk)
            sample_ns = novelty_score(sample_llk, sample_rec)

            # # Compute precision, recall, f1_score based on threshold
            # threshold = self.compute_threshold(cl)
            
            # #1 normal 0 novel 
            # y_hat = np.greater(sample_ns, threshold)

            # precision = precision_score(sample_y,y_hat)
            # f1 = f1_score(sample_y, y_hat)
            # recall = recall_score(sample_y, y_hat)
<<<<<<< HEAD
=======
>>>>>>> message
>>>>>>> temp


            # Compute AUROC for this class
            this_class_metrics = [
                roc_auc_score(sample_y, sample_llk),  # likelihood metric
                roc_auc_score(sample_y, sample_rec),  # reconstruction metric
<<<<<<< HEAD
=======
<<<<<<< e68b04d9643bf8aa75b53953df98d650ab4d948c
                roc_auc_score(sample_y, sample_ns),    # novelty score
                precision,
                f1,
                recall,
                threshold,
                density
=======
>>>>>>> temp
                roc_auc_score(sample_y, sample_ns)    # novelty score
                # precision,
                # f1,
                # recall,
                # threshold,
                # density
<<<<<<< HEAD
=======
>>>>>>> message
>>>>>>> temp

            ]
            oc_table.add_row([cl_idx] + this_class_metrics)

            all_metrics.append(this_class_metrics)



        # Compute average AUROC and print table
        all_metrics = np.array(all_metrics)
        avg_metrics = np.mean(all_metrics, axis=0)
        oc_table.add_row(['avg'] + list(avg_metrics))
        print(oc_table)

        # Save table
        with open(self.output_file, mode='w') as f:
            f.write(str(oc_table))

    
        
    def compute_normalizing_coefficients(self, cl):
        # type: (int) -> Tuple[float, float, float, float]
        """
        Computes normalizing coeffients for the computation of the Novelty score (Eq. 9-10).
        :param cl: the class to be considered normal.
        :return: a tuple of normalizing coefficients in the form (llk_min, llk_max, rec_min, rec_max).
        """
        dataset = self.dataset
        dataset.val(cl)
        loader = DataLoader(dataset)

        sample_llk = np.zeros(shape=(len(loader),))
        sample_rec = np.zeros(shape=(len(loader),))
        for i, (x, y) in enumerate(loader):
            x = x.to('cuda')
            x_r, z, z_dist, s, log_jacob_s = self.model(x)

            self.loss(x, x_r, z, z_dist,s, log_jacob_s)

            sample_llk[i] = - self.loss.autoregression_loss
            sample_rec[i] = - self.loss.reconstruction_loss

        return sample_llk.min(), sample_llk.max(), sample_rec.min(), sample_rec.max()




    @property
    def empty_table(self):
        # type: () -> PrettyTable
        """
        Sets up a nice ascii-art table to hold results.
        This table is suitable for the one-class setting.
        :return: table to be filled with auroc metrics.
        """
        table = PrettyTable()
<<<<<<< HEAD
=======
<<<<<<< e68b04d9643bf8aa75b53953df98d650ab4d948c
        table.field_names = ['Class', 'AUROC-LLK', 'AUROC-REC', 'AUROC-NS', 'Precision',
                'F1',
                'Recall',
                'Threshold','Density']
=======
>>>>>>> temp
        table.field_names = ['Class', 'AUROC-LLK', 'AUROC-REC', 'AUROC-NS'
                # , 'Precision',
                # 'F1',
                # 'Recall', 'Threshold','Density'
                ]
<<<<<<< HEAD
=======
>>>>>>> message
>>>>>>> temp
        table.float_format = '0.3'
        return table



# compute best threshold 
    def compute_threshold(self, cl):

        dataset = self.dataset
        dataset.val2(cl)

        loader = DataLoader(dataset)

        sample_score = np.zeros(shape=(len(loader),))
        ytrue = np.zeros(shape=(len(loader),))
        for i, (x, y) in enumerate(loader):
            x = x.to('cuda')
            x_r, z, z_dist, s, log_jacob_s = self.model(x)
            self.loss(x, x_r, z, z_dist,s, log_jacob_s)
            sample_score[i] = - self.loss.total_loss # large score -- normal
            ytrue[i] =y.numpy()

        best_e = 0
        best_f = 0
        best_e_ = 0
        best_f_ = 0

        # real label y  normal 0,  novel 1

        # predict score  sample_score 
        # predict label y_hat
        minS = sample_score.min() - 0.1
        maxS = sample_score.max() + 0.1

        for e in np.arange(minS, maxS, 0.1):

            y_hat = np.less(sample_score, e) #  normal 0  novel1
            # # TP Predict novel as novel y =1, y_hat =1
            true_positive = np.sum(np.logical_and(y_hat, ytrue))
            # # FP Predict normal as novel y = 0, y_hat = 1
            # false_positive = np.sum(np.logical_and(y_hat, logical_not(y)))
            # # PN Predict novel as normal y =1, y_hat = 0
            # false_negative = np.sum(np.logical_and(np.logical_not(y_hat),y))
            if true_positive > 0:

                f1 = f1_score(ytrue, y_hat)
                if f1 > best_f:
                    best_f = f1
                    best_e = e
                if f1 >= best_f_:
                    best_f_ = f1
                    best_e_ = e

        best_e = (best_e + best_e_) / 2.0

        print("Best e: ", best_e)
        return best_e
