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
from result_helpers.utils import *

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import math

# from result_helpers import metric_method as mm



class OneClassTestHelper(object):
    """
    Performs tests for one-class datasets (MNIST or CIFAR-10).
    """

    def __init__(self, dataset, model, score_normed, novel_ratio, lam, checkpoints_dir, output_file, device,batch_size,pretrained):
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
        self.device = device
        self.name = model.name
        self.bs = batch_size
        self.pretrained = pretrained
        # control novel ratio in test sets.
        self.novel_ratio = novel_ratio

        # normalized novelty score
        self.score_normed = score_normed
        # Set up loss function
        self.loss = SumLoss(self.model.name, lam = lam)

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
        bs =self.bs

        # Start iteration over classes
        for cl_idx, cl in enumerate(self.dataset.test_classes):

            # Load the checkpoint
            if self.pretrained:

                self.model.load_w(join(self.checkpoints_dir, f'{cl}{self.model.name}_ptr.pkl'))
            else:

                self.model.load_w(join(self.checkpoints_dir, f'{cl}{self.model.name}.pkl'))

            if self.score_normed:
                # we need a run on validation, to compute
                # normalizing coefficient of the Novelty Score (Eq.9 in LSA)
                min_llk, max_llk, min_rec, max_rec = self.compute_normalizing_coefficients(cl)

            # Prepare test set for one class and control the novel ratio in test set
            dataset = self.dataset
            
            dataset.test(cl,self.novel_ratio)
            data_num = len(dataset)

            loader = DataLoader(self.dataset, batch_size =bs)

            sample_llk = np.zeros(shape=(len(dataset),))
            sample_rec = np.zeros(shape=(len(dataset),))
            sample_y = np.zeros(shape=(len(dataset),))

            for i, (x, y) in tqdm(enumerate(loader), desc=f'Computing scores for {dataset}'):
                x = x.to(self.device)
                
                if self.name == 'LSA':
                    x_r = self.model(x)
                    self.loss.lsa(x, x_r,False)

                elif self.name == 'LSA_EN':
                    
                    x_r, z, z_dist = self.model(x)
                    self.loss.lsa_en(x, x_r, z, z_dist,False)
                
                elif self.name in ['LSA_SOS', 'LSA_MAF']:
                    x_r, z, s, log_jacob_T_inverse = self.model(x)
                    self.loss.lsa_flow(x,x_r,s,log_jacob_T_inverse,False)
                
                elif self.name in ['SOS', 'MAF','E_SOS','E_MAF']:
                    s, log_jacob_T_inverse = self.model(x)
                    self.loss.flow(s,log_jacob_T_inverse, False)
                
                elif self.name == 'EN':
                    z_dist = model(x)
                    self.loss.en(z_dist,False)

                
                
                sample_y[i*bs:i*bs+bs] = y
                # score larger-->normal data
                if self.name in ['LSA','LSA_MAF','LSA_SOS','LSA_EN']:
                    sample_rec[i*bs:i*bs+bs] = - self.loss.reconstruction_loss.cpu().numpy()
                    
                if self.name in ['LSA_MAF','LSA_SOS','LSA_EN','EN','SOS','MAF','E_SOS','E_MAF']:    
                    sample_llk[i*bs:i*bs+bs] = - self.loss.nllk.cpu().numpy()
                    # print (sample_llk[i])

            print(sample_llk)

            print(f"NAN_num:{sum(sample_llk!=sample_llk)}")

            # +inf,-inf,nan
            sample_llk = modify_inf(sample_llk)

            llk1= np.dot(sample_llk,sample_y).sum()
            llk2 = sample_llk.sum()-llk1

            # llk1 should be larger than llk2
            llk1 =llk1/np.sum(sample_y)
            # average llk for normal examples
            llk2 =llk2/(data_num-np.sum(sample_y)) 
            # average llk for novel examples

            if self.score_normed:
                print(f'min_llk:{min_llk},max_llk:{max_llk}'
                    )
                print(f'min_rec:{min_rec},max_rec:{max_rec}')

                # Normalize scores
                sample_llk = normalize(sample_llk, min_llk, max_llk)

                sample_rec = normalize(sample_rec, min_rec, max_rec)

            #print(sample_llk)
            # Compute the normalized novelty score

            # llk maybe too large or too small
            # why llk can be Nan?
            
            sample_llk[sample_llk==float('+inf')]= 10**35
            sample_llk[sample_llk==float('-inf')]= -10**35

            
            sample_ns = novelty_score(sample_llk, sample_rec)
            
            # Compute precision, recall, f1_score based on threshold
            # threshold = self.compute_threshold(cl)
            # y_hat = np.less(sample_ns, threshold)

            # precision = precision_score(sample_y,y_hat)
            # f1 = f1_score(sample_y, y_hat)
            # recall = recall_score(sample_y, y_hat)

            ## metrics 
            this_class_metrics = [
                roc_auc_score(sample_y, sample_ns)    #
            ]

            if self.name in ['LSA_EN','LSA_SOS','LSA_MAF']:
                this_class_metrics.append(
                roc_auc_score(sample_y, sample_llk))
                
                this_class_metrics.append(
                roc_auc_score(sample_y, sample_rec))

                this_class_metrics.append(llk1)
                this_class_metrics.append(llk2)
            # write on table
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
        bs = self.bs
        dataset = self.dataset
        dataset.val(cl)

        loader = DataLoader(dataset, batch_size= bs)

        sample_llk = np.zeros(shape=(len(dataset),))
        sample_rec = np.zeros(shape=(len(dataset),))
        
        for i, (x, y) in enumerate(loader):
            
            x = x.to(self.device)

            if self.name == 'LSA':
                x_r = self.model(x)
                self.loss.lsa(x, x_r,False)

            elif self.name == 'LSA_EN':
                
                x_r, z, z_dist = self.model(x)
                self.loss.lsa_en(x, x_r, z, z_dist,False)
            
            elif self.name in ['LSA_SOS', 'LSA_MAF']:
                x_r, z, s, log_jacob_T_inverse = self.model(x)
                self.loss.lsa_flow(x,x_r,s,log_jacob_T_inverse,False)
            
            elif self.name in ['SOS', 'MAF']:
                s, log_jacob_T_inverse = self.model(x)
                self.loss.flow(s,log_jacob_T_inverse,False)
            
            elif self.name == 'EN':
                z_dist = model(x)
                self.loss.en(z_dist,False)

            # score larger-->normal data
            if self.name in ['LSA','LSA_MAF','LSA_SOS','LSA_EN']:
                sample_rec[i*bs:i*bs+bs] = - self.loss.reconstruction_loss.cpu().numpy()
            
            if self.name in ['LSA_MAF','LSA_SOS','LSA_EN','EN','SOS','MAF']:    
                sample_llk[i*bs:i*bs+bs] = - self.loss.nllk.cpu().numpy()

            # sample_llk[sample_llk!=sample_llk] = 0
            # sample_llk[sample_llk==float('+inf')]= 10**35
            # sample_llk[sample_llk==float('-inf')]= -10**35
            
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
        if self.name in ['LSA_MAF','LSA_SOS','LSA_EN']:

            table.field_names = ['Class', 'AUROC-NS', 'AUROC-LLK', 'AUROC-REC','llk1','llk2'
                ]
        elif self.name in ['MAF','SOS','EN','LSA']:
            table.field_names = ['Class', 'AUROC-NS'
                ]
        
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
            x = x.to(self.device)

            if self.name == 'LSA':
                x_r = self.model(x)
                self.loss.lsa(x, x_r)

            elif self.name == 'LSA_EN':
                
                x_r, z, z_dist = self.model(x)
                self.loss.lsa_en(x, x_r, z, z_dist)
            
            elif self.name in ['LSA_SOS', 'LSA_MAF']:
                x_r, z, s, log_jacob_T_inverse = self.model(x)
                self.loss.lsa_flow(x,x_r,s,log_jacob_T_inverse)
            
            elif self.name in ['SOS', 'MAF']:
                s, log_jacob_T_inverse = self.model(x)
                self.loss.flow(s,log_jacob_T_inverse)
            
            elif self.name == 'EN':
                z_dist = model(x)
                self.loss.en(z_dist)

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
