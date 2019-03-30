from os.path import join
from typing import Tuple

import numpy as np
import torch
from prettytable import PrettyTable
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.base import OneClassDataset
from models.base import BaseModule
from models.loss_functions import SumLoss
from datasets.utils import novelty_score
from datasets.utils import normalize


class OneClassTestHelper(object):
    """
    Performs tests for one-class datasets (MNIST or CIFAR-10).
    """

    def __init__(self, dataset, model, score_normed, novel_ratio,checkpoints_dir, output_file):
        # type: (OneClassDataset, BaseModule, str, str) -> None
        """
        Class constructor.

        :param dataset: dataset class.
        :param novel_ratio: novel_ratio in test sets
        :param model: py-torch model to evaluate.
        :param checkpoints_dir: directory holding checkpoints for the model.
        :param output_file: text file where to save results.
        :param score_normed: 1 normalized the novelty score with valid set, 0: not normalized
        """
        self.dataset = dataset
        self.model = model
        self.checkpoints_dir = checkpoints_dir
        self.output_file = output_file

        # control novel ratio in test sets.
        self.novel_ratio = novel_ratio

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

            if self.score_normed == 1:
                # we need a run on validation, to compute
                # normalizing coefficient of the Novelty Score (Eq.9 in LSA)
                min_llk, max_llk, min_rec, max_rec = self.compute_normalizing_coefficients(cl)

            # Prepare test set for one class and control the novel ratio in test set
            self.dataset.test(cl,self.novel_ratio)
            loader = DataLoader(self.dataset)

            sample_llk = np.zeros(shape=(len(loader),))
            sample_rec = np.zeros(shape=(len(loader),))
            sample_y = np.zeros(shape=(len(loader),))

            for i, (x, y) in tqdm(enumerate(loader), desc=f'Computing scores for {self.dataset}'):
                x = x.to('cuda')
                x_r, z, z_dist, s, log_jacob_s = self.model(x)
                self.loss(x, x_r, z, z_dist,s,log_jacob_s)

                sample_llk[i] = - self.loss.autoregression_loss
                sample_rec[i] = - self.loss.reconstruction_loss
                sample_y[i] = y.item()

            if self.score_normed == 1:
                # Normalize scores
                sample_llk = normalize(sample_llk, min_llk, max_llk)

                sample_rec = normalize(sample_rec, min_rec, max_rec)

            # Compute the normalized novelty score
            sample_ns = novelty_score(sample_llk, sample_rec)
            
            # Compute AUROC for this class
            this_class_metrics = [
                roc_auc_score(sample_y, sample_llk),  # likelihood metric
                roc_auc_score(sample_y, sample_rec),  # reconstruction metric
                roc_auc_score(sample_y, sample_ns)    # novelty score
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
        self.dataset.val(cl)
        loader = DataLoader(self.dataset)

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
        table.field_names = ['Class', 'AUROC-LLK', 'AUROC-REC', 'AUROC-NS']
        table.float_format = '0.3'
        return table




#################For Train
class OneClassTrainHelper(object):
    """
    Performs tests for one-class datasets (MNIST or CIFAR-10).
    """

    def __init__(self, dataset, model, optimizer, checkpoints_dir,log_interval=1000, train_epoch=100):
        # type: (OneClassDataset, BaseModule, str, str) -> None
        """
        Class constructor.

        :param dataset: dataset class.
        :param model: pytorch model to train.
        :param checkpoints_dir: directory holding checkpoints for the model.
        :param output_file: text file where to save results.
        """
        self.dataset = dataset
        self.model = model
        # self.model_name = model_name

        self.checkpoints_dir = checkpoints_dir
        self.train_epoch = train_epoch
        self.optimizer = optimizer
        self.combine_density = combine_density

        # Set up loss function
        # if self.model.name =="SOSLSA":
        #     self.loss= SumLoss(self.model.name, lam = 0.1)
        self.loss = SumLoss(self.model.name)
        self.log_interval = log_interval


    def train_one_class_classification(self):
        # type: () -> None
        """
        Actually performs trains.
        """

        for cl_idx, cl in enumerate(self.dataset.train_classes):

            # Run the actual train
            self.dataset.train(cl)
            # load train_data in normal class
            loader = DataLoader(self.dataset, batch_size=100)

            sample_llk = np.zeros(shape=(len(loader),))
            sample_rec = np.zeros(shape=(len(loader),))
            sample_y = np.zeros(shape=(len(loader),))

            for epoch in range(self.train_epoch):

                self.model.train()
                epoch_loss = 0
                epoch_recloss = 0
                epoch_regloss = 0

                for batch_idx, (x, y) in tqdm(enumerate(loader), desc=f'Training models for {self.dataset}'):
                    # Clear grad for every batch
                    self.model.zero_grad()
                    # x_tra
                    x = x.to('cuda')

                    x_r, z, z_dist,s,log_jacob_s = self.model(x)

                    self.loss(x, x_r, z,z_dist, s, log_jacob_s) 

                    (self.loss.total_loss).backward()

                    self.optimizer.step()

                    epoch_loss=+self.loss.total_loss.item()
                    epoch_recloss =+ self.loss.reconstruction_loss.item()
                    epoch_regloss =+ self.loss.autoregression_loss.item()

                    # print batch result
                    
                print('Train Epoch: {} \tLoss: {:.6f}\tRec: {:.6f},Reg: {:.6f}'.format(
                            epoch, epoch_loss,epoch_recloss,epoch_regloss))

            #save model for every normal class
            print("Training finish! Normal_class:>>>>>",cl)
            print(join(self.checkpoints_dir,f'{cl}{self.model.name}.pkl'))
            torch.save(self.model.state_dict(), join(self.checkpoints_dir,f'{cl}{self.model.name}.pkl'))
       