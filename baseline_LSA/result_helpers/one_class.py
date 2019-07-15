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
from models.loss_functions import LSALoss
from utils import novelty_score
from utils import normalize
import torch.optim as optim
from torch import save

def load_checkpoint(model, model_dir):

    # load the checkpoint.
    checkpoint = torch.load(model_dir)
    # print('=> loaded checkpoint of {name} from {path}'.format(
    #     name=model.name, path=(path)
    # ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    epoch = checkpoint['epoch']
    return epoch

class OneClassResultHelper(object):
    """
    Performs tests for one-class datasets (MNIST or CIFAR-10).
    """

    def __init__(self, dataset, model, checkpoints_dir, output_file):
        # type: (OneClassDataset, BaseModule, str, str) -> None
        """
        Class constructor.

        :param dataset: dataset class.
        :param model: pytorch model to evaluate.
        :param checkpoints_dir: directory holding checkpoints for the model.
        :param output_file: text file where to save results.
        """
        self.dataset = dataset
        self.model = model
        self.checkpoints_dir = checkpoints_dir
        self.output_file = output_file

        # Set up loss function
        self.loss = LSALoss(cpd_channels=100)

    # @torch.no_grad()
    def test_one_class_classification(self):
        # type: () -> None
        """
        Actually performs tests.

        """

        # Prepare a table to show results
        oc_table = self.empty_table
        resume = False

        # Set up container for metrics from all classes
        all_metrics = []
    
        # Start iteration over classes
        for cl_idx, cl in enumerate(self.dataset.test_classes):

            # train model
            loss = LSALoss(cpd_channels= 100, lam= 0.1)
            optimizer = optim.Adam(self.model.parameters(), lr=10**-3)

            self.dataset.train(cl)
            trainloader = DataLoader(self.dataset, batch_size=256, shuffle=True, num_workers=1)

            print(f'Training {self.dataset.name}')
            
            if resume:
                epoch_start = load_checkpoint(self.model, f"{checkpoint_dir}{cl}.pkl")
            else:
                epoch_start = 1

            for epoch in range(epoch_start, 2000+1):  # loop over the dataset multiple times
                for i, (x, y) in enumerate(trainloader):
                  # get the inputs
                  x = x.to('cuda')

                  # zero the parameter gradients
                  optimizer.zero_grad()

                  # forward + backward + optimize
                  x_r, z, z_dist = self.model(x)
                  loss(x, x_r, z, z_dist).backward()
                  optimizer.step()

                # print statistics
                print('[%d] loss: %.3f' % (epoch + 1, loss.total_loss))

                if (epoch) % 100 == 0:
                    save(self.model.state_dict(), f'checkpoints/{self.dataset.name}/{cl}_{epoch}.pkl')

            # save(self.model.state_dict(), f'checkpoints/{self.dataset.name}/{cl}.pkl')
            save({'state': self.model.state_dict(), 'epoch': epoch},f'checkpoints/{self.dataset.name}/{cl}.pkl')
            
            print(f'Finished Training-{cl}')



            # Load the checkpoint
            # self.model.load_w(join(self.checkpoints_dir, f'{cl}.pkl'))
            load_checkpoint(self.model,join(self.checkpoints_dir, f'{cl}.pkl'))

            self.model.eval()
            # First we need a run on validation, to compute
            # normalizing coefficient of the Novelty Score (Eq.9)
            min_llk, max_llk, min_rec, max_rec = self.compute_normalizing_coefficients(cl)

            # Run the actual test
            self.dataset.test(cl)
            loader = DataLoader(self.dataset)

            sample_llk = np.zeros(shape=(len(loader),))
            sample_rec = np.zeros(shape=(len(loader),))
            sample_y = np.zeros(shape=(len(loader),))
            for i, (x, y) in tqdm(enumerate(loader), desc=f'Computing scores for {self.dataset}'):
                x = x.to('cuda')

                x_r, z, z_dist = self.model(x)

                self.loss(x, x_r, z, z_dist)

                sample_llk[i] = - self.loss.autoregression_loss
                sample_rec[i] = - self.loss.reconstruction_loss
                sample_y[i] = y.item()

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

            x_r, z, z_dist = self.model(x)

            self.loss(x, x_r, z, z_dist)

            sample_llk[i] = - self.loss.autoregression_loss
            sample_rec[i] = - self.loss.reconstruction_loss

        return sample_llk.min(), sample_llk.max(), sample_rec.min(), sample_rec.max()

    @property
    def empty_table(self,style):
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
