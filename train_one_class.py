################For Train
from os.path import join
from typing import Tuple

import copy
import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.base import OneClassDataset
from models.base import BaseModule
from models.loss_functions import SumLoss

import math

from utils import *
import torch.optim as optim

from tensorboardX import SummaryWriter
import flows_maf_models as maffnn
import flow_sos_models as sosfnn


class OneClassTrainHelper(object):
    """
    Performs tests for one-class datasets (MNIST or CIFAR-10).
    """


    def __init__(self, dataset, model, lr, lam, checkpoints_dir, device, kwargs, train_epoch=1000, batch_size = 100, before_log_epochs = 1000, pretrained= False):

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

        self.name = model.name

        self.checkpoints_dir = checkpoints_dir
        self.device = device
        self.kwargs = kwargs
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.before_log_epochs = before_log_epochs
        self.pretrained = pretrained


        self.lr = lr 
        self.lam = lam
        self.optimizer = optim.Adam(self.model.parameters(), lr= lr, weight_decay=1e-6)

        self.cl = self.dataset.normal_class

        # class for computing loss
        self.loss = SumLoss(self.model.name,lam = self.lam)


    def load_pretrained_model(self):
        # load pretrained model

        self.model.load_state_dict(torch.load(join(self.checkpoints_dir, f'{self.cl}LSA.pkl')),strict= False)

    def train_every_epoch(self, epoch):
            # global global_step, writer

            self.model.train()
            self.dataset.train(self.cl)

            epoch_loss = 0
            epoch_recloss = 0
            epoch_nllk = 0

            loader = DataLoader(self.dataset, batch_size = self.batch_size, **self.kwargs)

            epoch_size = self.dataset.length
            pbar = tqdm(total=epoch_size)

            for batch_idx, (x , y) in enumerate(loader):
                
                x = x.to(self.device)
                # Clear grad for every batch
                # self.model.zero_grad()
                self.optimizer.zero_grad()

                if self.name == 'LSA':
                    x_r = self.model(x)
                    self.loss.lsa(x, x_r)

                elif self.name == 'LSA_EN':
                    x_r, z, z_dist = self.model(x)
                    self.loss.lsa_en(x, x_r, z, z_dist)
                
                elif self.name in ['LSA_SOS', 'LSA_MAF']:
                    x_r, z, s, log_jacob_T_inverse = self.model(x)
                    self.loss.lsa_flow(x,x_r,s,log_jacob_T_inverse)
                
                elif self.name in ['SOS', 'MAF','E_MAF','E_SOS']:
                    s, log_jacob_T_inverse = self.model(x)
                    self.loss.flow(s, log_jacob_T_inverse)
                
                elif self.name == 'EN':
                    z_dist = model(x)
                    self.loss.en(z_dist)    



                epoch_loss = + self.loss.total_loss.item()*self.batch_size
                
                if self.name in ['LSA_EN','LSA_SOS','LSA_MAF']:
                    epoch_recloss =+ self.loss.reconstruction_loss.item()*self.batch_size
                    epoch_nllk = + self.loss.nllk.item()*self.batch_size

                # backward average loss along batch
                (self.loss.total_loss).backward()
                # update params
                self.optimizer.step()
                pbar.update(x.size(0))
                pbar.set_description('Train, Loss: {:.6f}'.format(epoch_loss / (pbar.n)))
        
                # writer.add_scalar('training/loss', loss.item(), global_step)
                # global_step += 1

            pbar.close()

            for module in self.model.modules():
                if isinstance(module, maffnn.BatchNormFlow):
                    module.momentum = 0
            for module in self.model.modules():
                if isinstance(module, maffnn.BatchNormFlow):
                    module.momentum = 1

                # print epoch result
            if self.name in ['LSA_EN','LSA_MAF','LSA_SOS']:

                print('Train Epoch: {}\tLoss: {:.6f}\tRec: {:.6f}\tNllk: {:.6f}'.format(
                            epoch, epoch_loss/epoch_size, epoch_recloss/epoch_size, epoch_nllk/epoch_size))
        
            else:
                print('Train Epoch-{}: {}\tLoss: {:.6f}'.format(
                            self.cl, epoch, epoch_loss/epoch_size))

    def validate(self, epoch, model, prefix = 'Validation'):

        model.eval()
        val_loss = 0
        val_nllk=0
        val_rec =0

        self.dataset.val(self.cl)

        loader = DataLoader(self.dataset, batch_size = self.batch_size,**self.kwargs)

        epoch_size = self.dataset.length
        batch_size = len(loader)

        pbar = tqdm(total=epoch_size)
        pbar.set_description('Eval')

        for batch_idx, (x,y) in enumerate(loader):
        
            x = x.to('cuda')
            pbar.update(x.size(0))

            with torch.no_grad():
                if self.name == 'LSA':
                    x_r = self.model(x)
                    self.loss.lsa(x, x_r,batch_average=False)

                elif self.name == 'LSA_EN':
                    x_r, z, z_dist = self.model(x)
                    self.loss.lsa_en(x, x_r, z, z_dist,batch_average=False)
                
                elif self.name in ['LSA_SOS', 'LSA_MAF']:
                    x_r, z, s, log_jacob_T_inverse = self.model(x)
                    self.loss.lsa_flow(x, x_r, s, log_jacob_T_inverse,batch_average=False)
                
                elif self.name in ['SOS', 'MAF','E_SOS','E_MAF']:
                    s, log_jacob_T_inverse = self.model(x)
                    self.loss.flow(s,log_jacob_T_inverse,batch_average=False)
                
                elif self.name == 'EN':
                    z_dist = model(x)
                    self.loss.en(z_dist,batch_average=False)

                val_loss += self.loss.total_loss.sum().item()
                if self.name in ['LSA_EN','LSA_MAF','LSA_SOS']:
                    val_nllk += self.loss.nllk.sum().item()
                    val_rec += self.loss.reconstruction_loss.sum().item()

                pbar.set_description('Val_loss: {:.6f}'.format(val_loss / pbar.n))
        
        pbar.close()
                
        if self.name in ['LSA_EN','LSA_MAF','LSA_SOS']:

            print('Nllk: {:.6f}\tRec: {:.6f}'.format(val_nllk/epoch_size, val_rec/epoch_size))
        
        return val_loss



    





    def train_one_class_classification(self):
        # type: () -> None
        """
        Actually performs trains.
        """     
        # writer = SummaryWriter(comment=self.name + "_" + self.dataset.name)
        # global_step = 0

        
        best_validation_epoch = 0
        best_validation_loss = float('inf')
        best_model = None  

        if self.pretrained:
            self.load_pretrained_model()

        for epoch in range(self.train_epoch):

            # adjust learning rate
            #adjust_learning_rate(self.optimizer, epoch, self.lr)
            # adjust lam
            # if epoch >= 30 and (epoch % 20 ==0):

            #     lam= min(lam*10, self.lam)  
            #     self.loss = SumLoss(self.model.name,self.lam)
            self.train_every_epoch(epoch)

            # validate
            validation_loss = self.validate(epoch, self.model)

            if epoch > self.before_log_epochs: # wait at least some epochs to log
               
                if (validation_loss < best_validation_loss):
                    best_validation_loss = validation_loss
                    best_validation_epoch = epoch
                    best_model = self.model 
                    
                    # if epoch % 10 == 0:
                    print(f'Best_epoch at :{best_validation_epoch} with valid_loss:{best_validation_loss}' )

                    torch.save(best_model.state_dict(), join(self.checkpoints_dir,f'{self.dataset.normal_class}{self.name}.pkl'))

            # converge?
            if (epoch - best_validation_epoch >= 30) and (best_validation_epoch > self.before_log_epochs+2): # converge? 
                    break
        print("Training finish! Normal_class:>>>>>",self.cl)
        
        print(join(self.checkpoints_dir,f'{self.cl}{best_model.name}.pkl'))

        torch.save(best_model.state_dict(), join(self.checkpoints_dir,f'{self.dataset.normal_class}{self.name}.pkl'))
        

