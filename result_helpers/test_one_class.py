"""
Train or Test.

Train and Test

"""

from os.path import join
from typing import Tuple
import torch.optim as optim
import numpy as np
import torch

from torch.utils.data import DataLoader


from datasets.base import OneClassDataset
from models.base import BaseModule
from models.loss_functions import *
from datasets.utils import novelty_score
from datasets.utils import normalize

from result_helpers.utils import *
from utils import *

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from models import LSA_MNIST
from models.loss_functions import *
import math

from prettytable import PrettyTable

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from models.flow_sos_models import BatchNormFlow
from scipy.stats import norm

import seaborn as sns


def _init_fn():
    np.random.seed(12)


class OneClassTestHelper(object):
    """
    Performs tests for one-class datasets (MNIST or CIFAR-10).
    """

    def __init__(
        self, dataset,
        model,
        score_normed,
        lam,
        checkpoints_dir,
        result_file_path,
        batch_size,
        lr,
        epochs,
        before_log_epochs,
        code_length,
        mulobj,
        test_checkpoint,
        log_step,
        device,
        fixed=False,
        pretrained=False,
        load_lsa=False):
        """
        Class constructor.

        :param dataset: dataset class.
        :param model: py-torch model to evaluate.
        :param score_normed: 1 normalized the novelty score with valid set, 0: not normalized
        :param checkpoints_dir: directory holding checkpoints for the model.
        :param result_file_path: text file where to save results.
        """

        self.dataset = dataset
        self.model = model
        self.device = device

        # use the same initialization for all models
        # torch.save(self.model.state_dict(), join(checkpoints_dir, f'{model.name}_start.pkl'))
        
        self.checkpoints_dir = checkpoints_dir
        self.log_step = log_step
        self.test_checkpoint = test_checkpoint
        self.result_file_path = result_file_path
        self.name = model.name
        self.batch_size = batch_size
        self.lam = lam

        # training-strategy
        if mulobj:
            if load_lsa:
                self.train_strategy = 'fixmul'
            else:
                self.train_strategy = 'mul'
        elif fixed:
            self.train_strategy = 'fix'
        elif pretrained:
            self.train_strategy = 'prt'
        else:
            self.train_strategy= f'{self.lam}'

        

        self.mulobj = mulobj# whether use mul-gradient
        self.score_normed = score_normed # normalized novelty score
        
        self.style = 'auroc' # table-style
        self.code_length = code_length

        self.fixed = fixed
        self.pretrained = pretrained
        self.load_lsa = load_lsa
        self.ae_finished = False

        self.optimizer = None
        self.ae_optimizer = None
        self.est_optimizer = None

        if self.fixed or self.pretrained:
            if (self.model.estimator !=None):
                self.est_optimizer = optim.Adam(self.model.estimator.parameters(),lr = lr, weight_decay = 1e-6)
            self.ae_optimizer  = optim.Adam(list(self.model.encoder.parameters())+list(self.model.decoder.parameters()),lr = lr, weight_decay = 1e-6)

        self.optimizer = optim.Adam(self.model.parameters(), lr= lr, weight_decay=1e-6)

        ## Set up loss function
        # encoder + decoder
        if self.name in ['LSA','LSAD','LSAW']:
            self.loss = LSALoss(cpd_channels=200)
        # encoder + estimator+ decoder
        elif self.name in ['LSA_EN','LSAW_EN']:
            self.loss = LSAENLoss(cpd_channels=200,lam=lam)
        
        elif self.name in ['LSA_SOS', 'LSA_MAF','LSAD_SOS','LSAW_SOS']:
            self.loss =LSASOSLoss(lam)

        elif self.name == 'SOS':
            self.loss = SOSLoss()
        else:
            ValueError("Wrong Model Name")
        
        print (f"Testing on {self.name}")

        # initialize dir
        self.model_dir = None
        self.best_model_dir = None
        self.result_dir = None

        # Related to training
       
        self.lr = lr
        self.train_epochs = epochs
        self.before_log_epochs = before_log_epochs
        
        
    def get_path(self, cl):
        name    = self.name 
        checkpoints_dir = self.checkpoints_dir
        test_checkpoint = self.test_checkpoint 
        lam     = self.lam 
        train_strategy = self.train_strategy

        self.model_dir = join(checkpoints_dir,f'{cl}{name}_{train_strategy}.pkl')
        self.model_detail_dir = join(checkpoints_dir,f'{cl}{name}_{train_strategy}_detail.pkl')
        self.best_model_dir = join(checkpoints_dir,f'{cl}{name}_{train_strategy}_b.pkl')
        self.best_model_rec_dir = join(checkpoints_dir,f'{cl}{name}_{train_strategy}_rec.pkl')
        self.best_model_rec_detail_dir = join(checkpoints_dir,f'{cl}{name}_{train_strategy}_rec_detail.pkl')

        
        # self.train_result_dir = join(checkpoints_dir,f'{cl}{name}_{train_strategy}_history_train')
        # self.valid_result_dir = join(checkpoints_dir,f'{cl}{name}_{train_strategy}_history_valid')
        # self.test_result_dir = join(checkpoints_dir,f'{cl}{name}_{train_strategy}_history_test')

        self.train_history_dir = join(checkpoints_dir,f'{cl}{name}_{train_strategy}_loss_history')


    def _eval_quantile(self, s):

    #  from s~N(R^d) to u~(0,1)^d
    #  u_i = phi(s_i), where phi is the cdf of N(0,1)
    # Compute S 
    # only for SOS density estimator

        bs = s.shape[0]
        s_dim = s.shape[1]
        s_numpy = s.cpu().numpy()
        q1 = []
        q2 = []
        qinf = []
        u_s = np.zeros((bs , s_dim))
        for i in range(bs):
            # for every sample
            # cdf
            u_si = norm.cdf(s_numpy[i, :])
            u_s[i,:] = u_si
            # Source point in the source uniform distribution
            # u = abs(np.ones((1,s_dim))*0.5-u_s)

            u = abs(0.5 - u_si)

            uq_1 = np.linalg.norm(u, 1)
            uq_2 = np.linalg.norm(u)
            uq_inf = np.linalg.norm(u, np.inf)

            q1.append(-uq_1)
            q2.append(-uq_2)
            qinf.append(-uq_inf)

        return q1, q2, qinf, u_s

    def _eval(self, x, average=True, quantile_flag=False):

        if self.name in ['LSA','LSAD','LSAW']:
            # ok
            x_r = self.model(x)
            tot_loss = self.loss(x, x_r, average)

        elif self.name in ['LSA_EN','LSAW_EN']:
            x_r, z, z_dist = self.model(x)
            tot_loss = self.loss(x, x_r, z, z_dist, average)

        elif self.name in ['LSA_SOS','LSA_MAF','LSAD_SOS','LSAW_SOS','LSAW_MAF']:

            x_r, z, s, log_jacob_T_inverse = self.model(x)
            tot_loss = self.loss(x, x_r, s, log_jacob_T_inverse, average)
        elif self.name in 'SOS':
            s, log_jacob_T_inverse = self.model(x)
            tot_loss = self.loss(s, log_jacob_T_inverse,average)

        if quantile_flag:
            q1, q2, qinf, u_s = self._eval_quantile(s)
            return tot_loss, q1, q2, qinf, u_s
        else:
            return tot_loss 



    def train_every_epoch(self, epoch, cl):    
        # print(epoch)
        # print("weight")
        # print(self.model.encoder.conv[0].bn1b.weight.detach().cpu().numpy())
        # print(self.model.encoder.conv[0].bn1b.running_mean.detach().cpu().numpy())

        # model_copy = type(self.model)(input_shape= self.dataset.shape, code_length= 64, num_blocks =1, hidden_size= 2048, est_name = 'SOS') # get a new instance
        # model_copy.load_state_dict(self.model.state_dict())
        # model_copy.cuda()

        epoch_loss = 0
        epoch_recloss = 0
        epoch_nllk = 0
        
        bs      = self.batch_size

        self.dataset.train(cl)
        # self.model.train()
        loader = DataLoader(self.dataset, batch_size = self.batch_size, shuffle = False, worker_init_fn = _init_fn, num_workers = 0)
        
        dataset_name = self.dataset.name
        epoch_size = self.dataset.length
        # pbar = tqdm(total=epoch_size)
        s_alpha = 0

        for batch_idx, (x , _) in enumerate(loader):

            x = x.to(self.device)
            self.optimizer.zero_grad()
            self._eval(x)

            # backward average loss along batch
            if (self.mulobj):
            # Multi-objective Optimization
                # g1: the gradient of reconstruction loss w.r.t the parameters of encoder
                # g2: the gradient of auto-regression loss w.r.t the parameters of encoder
                if self.name in ['LSA_SOS','LSA_MAF']:
                    if epoch < 10:
                        lr = 0.0001
                    elif epoch < 1000:
                        lr = 0.00001
                    else:
                        lr = 0.000005
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr

                # Backward Total loss= Reconstruction loss + Auto-regression Loss
                torch.autograd.backward(self.loss.total_loss, self.model.parameters(),retain_graph =True)
                # g1_list = g1 + g2
                g1_list = [pi.grad.data.clone() for pi in list(self.model.encoder.parameters())]    
                # Backward Auto-regression Loss, the gradients computed in the first backward are not cleared
                # torch.autograd.backward(self.loss.autoregression_loss, self.model.parameters())

                # for p in self.model.estimator.parameters():
                #     print(p.grad.data[0])
                #     break

                for p in self.model.estimator.parameters():
                    p.grad.zero_()
                    # print (p.grad)
                
                # torch.autograd.backward(self.loss.autoregression_loss, self.model.encoder.parameters())
                torch.autograd.backward(self.loss.autoregression_loss, list(self.model.encoder.parameters())+list(self.model.estimator.parameters()))

                # for p in self.model.estimator.parameters():
                #     print(p.grad.data[0])
                #     break
                
                #g2_list = g1_list + g2
                g2_list = [pi.grad.data.clone() for pi in list(self.model.encoder.parameters())]
                
                # the gradients w.r.t estimator are accumulated, div 2 to get the original one
                # compute alpha
                top = 0
                down = 0
                # the formula (4) in Multi_Task Learning as Multi-Objective Function
                for i in range(len(g2_list)):
                    g2 = (g2_list[i] - g1_list[i])  # regression loss
                    g1 = (g1_list[i] - g2)  # reconstruction loss
                    g1 = g1.view(-1,)
                    g2 = g2.view(-1,)
                    top = top + torch.dot((g2 - g1), g2)
                    down = down + torch.dot((g1 - g2), (g1 - g2))

                alpha = (top / down).item()
                alpha = max(min(alpha, 1), 0)

                # compute new gradient of Shared Encoder by combined two gradients
                i = 0
                s_alpha = s_alpha + alpha * x.shape[0]

                for p in self.model.encoder.parameters():
                    newlrg2 = g2_list[i] - g1_list[i]
                    newlrg1 = g1_list[i] - newlrg2
                    # compute the multi-gradient of the parameters in the encoder
                    p.grad.data = torch.mul(newlrg1, alpha) + torch.mul(newlrg2, 1 - alpha)
                    i = i + 1

                self.optimizer.step()

            elif self.fixed:
                if (not self.ae_finished):
                    self.loss.reconstruction_loss.backward()
                    self.ae_optimizer.step()
                else:
                    self.loss.autoregression_loss.backward()
                    self.est_optimizer.step()
                    # print("weight")
                    # print(self.model.encoder.fc[0].weight)
                    # print(self.loss.reconstruction_loss.item())

            elif self.pretrained:
                if (not self.ae_finished):
                    self.loss.reconstruction_loss.backward()
                    self.ae_optimizer.step()
                else:
                    self.loss.total_loss.backward()
                    self.optimizer.step()
            else:
                self.loss.total_loss.backward()
                self.optimizer.step()

            # def compare_models(model_1, model_2):
            #     models_differ = 0
            #     for key_item_1, key_item_2 in zip(model_1.encoder.state_dict().items(), model_2.encoder.state_dict().items()):
            #         if torch.equal(key_item_1[1], key_item_2[1]):
            #             pass
            #         else:
            #             models_differ += 1
            #             if (key_item_1[0] == key_item_2[0]):
            #                 print('Mismtach found at', key_item_1[0])
            #             else:
            #                 raise Exception
            #     if models_differ == 0:
            #         print('Models match perfectly! :)')
            #     else:
            #         print("Mismtach")

            # if batch_idx ==1:
            #     compare_models(self.model, model_copy)
            
            # print(f"{batch_idx}_running mean:{self.model.encoder.conv[0].bn1b.running_mean.detach().cpu().numpy()}")


            epoch_loss +=  self.loss.total_loss.item()*x.shape[0]

            if self.name in ['LSA_EN','LSAW_EN','LSA_SOS','LSAD_SOS','LSAW_SOS','LSA_MAF']:
                # print(self.loss.reconstruction_loss)
                # print(self.loss.autoregression_loss)
                epoch_recloss += self.loss.reconstruction_loss.item()*x.shape[0]
                epoch_nllk +=  self.loss.autoregression_loss.item()*x.shape[0]


            # pbar.update(x.size(0))
            # pbar.set_description('Train, Loss: {:.6f}'.format(epoch_loss / (pbar.n)))

            
            # images
            # plot_source_dist_by_dimensions(sample_u, sample_y, f"{self.train_result_dir}_{epoch}")

        # pbar.close()
        # if (epoch % 200 ==0) and (self.name in ['LSA_SOS','SOS']):
        #     self.train_validate(epoch,loader,bs)

        # print epoch result
        # if self.name in ['LSA_SOS']:
            
        #     print('{}Train Epoch-{}: {}\tLoss: {:.6f}\tRec: {:.6f}\tNllk: {:.6f}\tNllk1: {:.6f}\tNjob: {:.6f}'.format(self.name,
        #             self.dataset.normal_class, epoch, epoch_loss/epoch_size, epoch_recloss/epoch_size, epoch_nllk/epoch_size, epoch_nllk1/epoch_size, epoch_njob/epoch_size))


        if self.name in ['LSA_EN','LSAW_EN','LSA_SOS','LSA_MAF','LSAD_SOS','LSAW_SOS']:

            print('{}Train Epoch-{}: {}\tLoss: {:.6f}\tRec: {:.6f}\tNllk:{:.6f}\t'.format(self.name,
                    self.dataset.normal_class, epoch, epoch_loss/epoch_size, epoch_recloss/epoch_size, epoch_nllk/epoch_size))

        else:
            print('Train Epoch-{}: {}\tLoss:{:.6f}\t'.format(
                    self.dataset.normal_class, epoch, epoch_loss/epoch_size))

        if self.mulobj:
            print (f'Adaptive Alpha:{s_alpha/epoch_size}')

        # for module in self.model.modules():
        #     if isinstance(module, BatchNormFlow):
        #         module.momentum = 0
        # with torch.no_grad():
        #     self.model(tempx)
        # for module in self.model.modules():
        #     if isinstance(module, BatchNormFlow):
        #         module.momentum = 1


        return epoch_loss/epoch_size, epoch_recloss/epoch_size, epoch_nllk/epoch_size

    def validate(self, epoch, cl):

        prefix = 'Validation'
        
        val_loss = 0
        val_nllk = 0
        val_rec  = 0
        bs = self.batch_size

        self.dataset.val(cl)
        loader = DataLoader(self.dataset, bs, shuffle = False)

        epoch_size = self.dataset.length
        batch_size = len(loader)

        # if (epoch % 200 ==0) and (self.name in ['LSA_SOS','SOS']):
        #          # density related 
        #         sample_llk = np.zeros(shape=(len(self.dataset),))
        #         sample_nrec = np.zeros(shape=(len(self.dataset),))
                
        #         # quantile related 
        #         sample_q1 = np.zeros(shape=(len(self.dataset),))
        #         sample_q2 = np.zeros(shape=(len(self.dataset),))
        #         sample_qinf = np.zeros(shape=(len(self.dataset),))
        #         # source point u (64)
        #         sample_u = np.zeros(shape=(len(self.dataset), self.code_length))
        #         # true label
        #         sample_y = np.zeros(shape=(len(self.dataset),))
        

        for batch_idx, (x,y) in enumerate(loader):
        
            x = x.to(self.device)

            with torch.no_grad():
                
                # record details 
                # if (epoch % 200 ==0) and (self.name in ['LSA_SOS','SOS']):
                #     #new add!!!
                #     _, q1,q2, qinf, u= self._eval(x, average= False, quantile_flag= True)

                    # quantile 
                    
                    # i = batch_idx
                    # sample_q1[i*bs:i*bs+bs] = q1
                    # sample_q2[i*bs:i*bs+bs] = q2
                    # sample_qinf[i*bs:i*bs+bs] = qinf
                    
                    # # source point 
                    # sample_u[i*bs:i*bs+bs] = u

                    # sample_y[i*bs:i*bs+bs] = y
                    #     # score larger-->normal data
                    # sample_nrec[i*bs:i*bs+bs] = - self.loss.reconstruction_loss.cpu().numpy()
                            
                    # sample_llk[i*bs:i*bs+bs] = - self.loss.autoregression_loss.cpu().numpy()
                # else:
                loss =self. _eval(x, average= False)

                if self.name in ['LSA_EN','LSAW_EN','LSA_SOS','LSA_MAF','LSAD_SOS','LSAW_SOS']:
                    val_nllk += self.loss.autoregression_loss.sum().item()
                    val_rec += self.loss.reconstruction_loss.sum().item()
                    val_loss = val_nllk + val_rec
                else:
                     val_loss += self.loss.total_loss.sum().item()

                 


        # if (epoch % 200 == 0) and (self.name in ['LSA_SOS','SOS']):
        #     # save test-historty
        #     np.savez(f"{self.valid_result_dir}_{epoch}", 
        #             sample_y = sample_y, 
        #             sample_nrec = sample_nrec, 
        #             sample_llk = sample_llk,
        #             sample_qinf = sample_qinf,
        #             sample_u = sample_u)
            

                                    

        if self.name in ['LSA_EN','LSAW_EN','LSA_SOS','LSA_MAF','LSAD_SOS','LSAW_SOS']:
            print('Val_loss:{:.6f}\t Rec: {:.6f}\t Nllk: {:.6f}'.format(val_loss/epoch_size, val_rec/epoch_size, val_nllk/epoch_size))
        else:
            print('Val_loss:{:.6f}\t'.format(val_loss/epoch_size))

        return val_loss/epoch_size, val_rec/epoch_size,val_nllk/epoch_size

    def train_validate(self, epoch, loader, bs):
        
        self.model.eval()
        sample_llk = np.zeros(shape=(len(self.dataset),))
        sample_nrec = np.zeros(shape=(len(self.dataset),))
        
        # quantile related 
        sample_q1 = np.zeros(shape=(len(self.dataset),))
        sample_q2 = np.zeros(shape=(len(self.dataset),))
        sample_qinf = np.zeros(shape=(len(self.dataset),))
        # source point u (64)
        sample_u = np.zeros(shape=(len(self.dataset), self.code_length))
        # true label
        sample_y = np.zeros(shape=(len(self.dataset),))

        for batch_idx, (x,y) in enumerate(loader):
            x = x.to(self.device)
            with torch.no_grad():
                
                    # record details 
                #new add!!!
                _, q1,q2, qinf, u= self._eval(x, average= False, quantile_flag= True)

                # quantile 
                
                i = batch_idx
                sample_q1[i*bs:i*bs+bs] = q1
                sample_q2[i*bs:i*bs+bs] = q2
                sample_qinf[i*bs:i*bs+bs] = qinf
                
                # source point 
                sample_u[i*bs:i*bs+bs] = u

                sample_y[i*bs:i*bs+bs] = y
                    # score larger-->normal data
                sample_nrec[i*bs:i*bs+bs] = - self.loss.reconstruction_loss.cpu().numpy()
                        
                sample_llk[i*bs:i*bs+bs] = - self.loss.autoregression_loss.cpu().numpy()

        np.savez(f"{self.train_result_dir}_{epoch}", 
                    sample_y = sample_y, 
                    sample_nrec = sample_nrec, 
                    sample_llk = sample_llk,
                    sample_qinf = sample_qinf,
                    sample_u = sample_u)


    def train_one_class_classification(self, cl):
        # type: () -> None
        """
        Actually performs trains.
        """

        self.get_path(cl)

        best_validation_epoch = 0

        best_train_epoch = 0
        best_train_rec_epoch = 0

        best_validation_loss = float('+inf')
        best_validation_rec = float('+inf')
        best_validation_nllk = float('+inf')

        best_train_loss = float('+inf')
        best_train_rec = float('+inf')
        best_train_nllk = float('+inf')

        best_model = None
        best_rec_model = None

        old_validation_loss = float('+inf')
        loss_history = {}
        loss_history['train_loss'] = []
        loss_history['train_rec'] = []
        loss_history['train_nllk'] = []

        loss_history['validation_loss'] = []
        loss_history['validation_rec'] = []
        loss_history['validation_nllk'] = []

        print(f"n_parameters:{self.model.n_parameters}")
        
        converge_epochs = 100


        for epoch in range(self.train_epochs):
            
            model_dir_epoch = join(self.checkpoints_dir,f'{cl}{self.name}_{self.train_strategy}_{epoch}.pkl')
            
            #train every epoch
            self.model.train()
            if self.load_lsa:
                if self.name in ['LSAD_SOS','LSAD_EN']:
                    model_style = 'LSAD'
                elif self.name in ['LSAW_SOS','LSAW_EN']:
                    model_style = 'LSAW'
                else:
                    model_style = 'LSA'

                self.model.load_lsa(join(self.checkpoints_dir,f'{cl}{model_style}_1_b.pkl'))
                print("load pre-traind autoencoder")
                self.ae_finished = True # Start from pretrained autoencoder

            train_loss, train_rec, train_nllk = self.train_every_epoch(epoch,cl) 
            # train_loss =0; train_rec = 0; train_nllk =0;    

            # validate every epoch
            self.model.eval()
            validation_loss, validation_rec, validation_nllk = self.validate(epoch, cl)

            loss_history['train_loss'].append(train_loss)
            loss_history['train_rec'].append(train_rec)
            loss_history['train_nllk'].append(train_nllk)

            loss_history['validation_loss'].append(validation_loss)
            loss_history['validation_rec'].append(validation_rec)
            loss_history['validation_nllk'].append(validation_nllk)

               
            if (validation_loss < best_validation_loss): 
                best_validation_loss = validation_loss
                best_validation_epoch = epoch
                best_model = self.model
                if epoch > self.before_log_epochs:
                    torch.save(best_model.state_dict(), self.best_model_dir)

                print(f'Best_valid_epoch at :{epoch} with valid_loss:{best_validation_loss}' )

            if (epoch % self.log_step == 0 ) :
                    torch.save(self.model.state_dict(), model_dir_epoch)
            
            # early stop
            if (epoch - best_validation_epoch)> converge_epochs and (epoch > self.before_log_epochs):
                break

            



            
        print("Training finish! Normal_class:>>>>>", cl)
        
        torch.save(self.model.state_dict(), self.model_dir)
        
        state = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
             'optimizer': self.optimizer.state_dict(),'ae_optimizer':self.ae_optimizer,'est_optimizer':self.est_optimizer}
        
        torch.save(state, self.model_detail_dir)
        
        np.savez(self.train_history_dir, loss_history= loss_history, best_validation_epoch = best_validation_epoch, best_validation_rec_epoch= best_train_rec_epoch)

    def test_one_class_classification(self, cl):
        """
        Test fore one class.

        cl as the nominal class, others classes as novel classes
        """
        # for specific checkpoint
        if (self.test_checkpoint is not None):
            # select one epoch to test
            checkpoint_name =\
                f'{cl}{self.name}_{self.train_strategy}_{self.test_checkpoint}'
            self.model_dir = join(self.checkpoints_dir,
                                  f'{checkpoint_name}.pkl')
            self.test_result_dir = join(self.checkpoints_dir,
                                        f'{checkpoint_name}_history_test')

        # load the checkpoint
        bs = self.batch_size
        self.model.load_w(self.model_dir)
        print(f"Load Model from {self.model_dir}")

        self.model.eval()
        # Test sets
        self.dataset.test(cl)

        loader = DataLoader(self.dataset, batch_size=bs, shuffle=False)

        # density related
        sample_llk = np.zeros(shape=(len(self.dataset),))
        sample_nrec = np.zeros(shape=(len(self.dataset),))
        # true label
        sample_y = np.zeros(shape=(len(self.dataset),))
        # quantile related
        sample_q1 = np.zeros(shape=(len(self.dataset),))
        sample_q2 = np.zeros(shape=(len(self.dataset),))
        sample_qinf = np.zeros(shape=(len(self.dataset),))
        # source distribution u (64)
        sample_u = np.zeros(shape=(len(self.dataset), self.code_length))

        for i, (x, y) in tqdm(enumerate(loader),
                              desc=f'Computing scores for {self.dataset}'):
            x = x.to(self.device)
            with torch.no_grad():
                if self.name in ['LSA_SOS', 'SOS', 'LSA_MAF',
                                 'LSAD_SOS', 'LSAW_SOS']:
                    tot_loss, q1, q2, qinf, u =\
                        self._eval(x, average=False, quantile_flag=True)
                    # quantile
                    sample_q1[i * bs:i * bs + bs] = q1
                    sample_q2[i * bs:i * bs + bs] = q2
                    sample_qinf[i * bs:i * bs + bs] = qinf
                    # source point
                    sample_u[i * bs:i * bs + bs] = u
                else:
                    tot_loss = self._eval(x, average=False)

            # True label
            sample_y[i * bs:i * bs + bs] = y
            # score larger-->normal data
            if self.name in ['LSA', 'LSAD', 'LSAW',
                             'LSA_SOS', 'LSAD_SOS', 'LSAW_SOS',
                             'LSA_EN', 'LSA_MAF']:
                sample_nrec[i * bs:i * bs + bs]\
                    = - self.loss.reconstruction_loss.cpu().numpy()

            if self.name in ['LSA_SOS', 'LSAD_SOS', 'LSAW_SOS',
                             'LSA_EN',
                             'EN', 'SOS', 'LSA_MAF']:
                sample_llk[i * bs:i * bs + bs]\
                    = - self.loss.autoregression_loss.cpu().numpy()

        sample_llk = modify_inf(sample_llk)

        if self.score_normed:
            # Normalize scores
            # normalizing coefficient of the Novelty Score (Eq.9 in LSA)
            min_llk, max_llk, min_rec, max_rec,min_q1,max_q1,min_q2,max_q2,min_qinf,max_qinf = self.compute_normalizing_coefficients(cl)
            sample_llk = normalize(sample_llk, min_llk, max_llk)
            sample_nrec = normalize(sample_nrec, min_rec, max_rec)
            sample_q1 = normalize(sample_q1, min_q1, max_q1)
            sample_q2 = normalize(sample_q2, min_q2, max_q2)
            sample_q1 = normalize(sample_qinf, min_qinf, max_qinf)


        sample_ns = novelty_score(sample_llk, sample_nrec)

        # if self.name in ['LSA_SOS', 'LSAD_SOS', 'LSAW_SOS']:
        #     sample_ns_t = sample_llk # larger, normal
        # else:
        sample_ns_t = sample_ns  # larger, normal

        # # based on quantile-norm-inf
        if self.name in ['LSA_SOS','LSA_MAF','LSAD_SOS','LSAW_SOS']:

            precision_den, f1_den, recall_den = compute_density_metric(self.name, sample_ns_t,sample_y)
            precision_q1, f1_q1, recall_q1 = compute_quantile_metric(self.name, sample_q1, sample_y, self.code_length, '1')
            precision_q2, f1_q2, recall_q2 = compute_quantile_metric(self.name, sample_q2, sample_y, self.code_length, '2')
            precision_qinf, f1_qinf, recall_qinf = compute_quantile_metric(self.name, sample_qinf, sample_y, self.code_length, 'inf')

            this_class_metrics = [roc_auc_score(sample_y, sample_ns),
                                  roc_auc_score(sample_y, sample_llk),
                                  roc_auc_score(sample_y, sample_nrec),
                                  roc_auc_score(sample_y, sample_q1),
                                  roc_auc_score(sample_y, sample_q2),
                                  roc_auc_score(sample_y, sample_qinf),    #
                                  precision_den,
                                  f1_den,
                                  recall_den,
                                  precision_q1,
                                  f1_q1,
                                  recall_q1,
                                  precision_q2,
                                  f1_q2,
                                  recall_q2,
                                  precision_qinf,
                                  f1_qinf,
                                  recall_qinf]
        elif self.name in ['LSA_EN']:
         # every row
            precision_den, f1_den, recall_den = compute_density_metric(self.name, sample_ns_t,sample_y)
            this_class_metrics = [
            roc_auc_score(sample_y, sample_ns),
            roc_auc_score(sample_y, sample_llk),
            roc_auc_score(sample_y, sample_nrec),
            precision_den,
            f1_den,
            recall_den
            ]
        elif self.name in ['LSA','LSAD','LSAW']:
        # every row
            this_class_metrics = [
            roc_auc_score(sample_y, sample_ns),
            ]
        elif self.name in ['SOS']:
        # every row
            precision_den, f1_den, recall_den = compute_density_metric(self.name, sample_ns_t,sample_y)
            precision_q1, f1_q1, recall_q1 = compute_quantile_metric(self.name, sample_q1, sample_y, self.code_length, '1')
            precision_q2, f1_q2, recall_q2 = compute_quantile_metric(self.name, sample_q2, sample_y, self.code_length, '2')
            precision_qinf, f1_qinf, recall_qinf = compute_quantile_metric(self.name, sample_qinf, sample_y, self.code_length, 'inf')

            this_class_metrics = [roc_auc_score(sample_y, sample_ns),
                                  precision_den,
                                  f1_den,
                                  recall_den,
                                    precision_q1,
                                    f1_q1,
                                    recall_q1,
                                    precision_q2,
                                    f1_q2,
                                    recall_q2,
                                    precision_qinf,
                                    f1_qinf,
                                    recall_qinf
                ]
        return this_class_metrics

    def test_classification(self):
        """
        Test Result.

        Test for all classes and generate result table
        """
        auroc_table = self.empty_table
        all_metrics = []

        # Start iteration over classes
        for cl_idx, cl in enumerate(self.dataset.test_classes):
            self.cl = cl
            self.get_path(cl)
            one_class_metric = self.test_one_class_classification(cl)
            auroc_table.add_row([cl_idx] + one_class_metric)
            all_metrics.append(one_class_metric)

        all_metrics = np.array(all_metrics)
        avg_metrics = np.mean(all_metrics, axis=0)

        auroc_table.add_row(['avg'] + list(avg_metrics))
        print(auroc_table)

        # Save table
        with open(self.result_file_path, mode='w') as f:
            f.write(str(auroc_table))



     
    def compute_normalizing_coefficients(self, cl):
        # type: (int) -> Tuple[float, float, float, float]
        """
        Computes normalizing coeffients for the computation of the Novelty score (Eq. 9-10).
        :param cl: the class to be considered normal.
        :return: a tuple of normalizing coefficients in the form (llk_min, llk_max, rec_min, rec_max).
        """
        bs = self.batch_size
        self.dataset.val(cl)
        loader = DataLoader(self.dataset, batch_size= bs, shuffle = False)

        sample_llk = np.zeros(shape=(len(self.dataset),))
        sample_nrec = np.zeros(shape=(len(self.dataset),))
        sample_q1 = np.zeros(shape=(len(self.dataset),))
        sample_q2 = np.zeros(shape=(len(self.dataset),))
        sample_qinf = np.zeros(shape=(len(self.dataset),))

        for i, (x, y) in enumerate(loader): 
            x = x.cuda()
            with torch.no_grad():

                if self.name in ['LSA_SOS','SOS','LSAD_SOS','LSAW_SOS']:
                    tot_loss, q1,q2,qinf,_ = self._eval(x, quantile_flag= True)
                    sample_q1[i*bs:i*bs+bs] = q1
                    sample_q2[i*bs:i*bs+bs] = q2
                    sample_qinf[i*bs:i*bs+bs] = qinf
                else:
                    tot_loss = self._eval( x, average = False)

            # score larger-->normal data
            if self.name in ['LSA','LSAD','LSAW','LSA_SOS','LSA_EN','LSAW_EN','LSAD_SOS','LSAW_SOS']:
                sample_nrec[i*bs:i*bs+bs] = - self.loss.reconstruction_loss.cpu().numpy()
        
            if self.name in ['LSA_SOS','LSA_EN','LSAW_EN','EN','SOS','LSAD_SOS','LSAW_SOS']:    
                sample_llk[i*bs:i*bs+bs] = - self.loss.autoregression_loss.cpu().numpy()

            
            
            sample_llk = modify_inf(sample_llk)

        return sample_llk.min(), sample_llk.max(), sample_nrec.min(), sample_nrec.max(),sample_q1.min(),sample_q1.max(),sample_q2.min(),sample_q2.max(),sample_qinf.min(), sample_qinf.max()

    @property
    def empty_table(self):
        # type: () -> PrettyTable
        """
        Sets up a nice ascii-art table to hold results.
        This table is suitable for the one-class setting.
        :return: table to be filled with auroc metrics.
        """
        table = PrettyTable()
        table.field_names = {

        'LSA_SOS':['Class', 'AUROC-NS', 'AUROC-LLK', 'AUROC-REC','AUROC-q1','AUROC-q2','AUROC-qinf','PRCISION','F1','RECALL', 'precision_q1', 'f1_q1','recall_q1','precision_q2', 'f1_q2','recall_q2','precision_qinf', 'f1_qinf','recall_qinf'],

        'LSAD_SOS':['Class', 'AUROC-NS', 'AUROC-LLK', 'AUROC-REC','AUROC-q1','AUROC-q2','AUROC-qinf','PRCISION','F1','RECALL', 'precision_q1', 'f1_q1','recall_q1','precision_q2', 'f1_q2','recall_q2','precision_qinf', 'f1_qinf','recall_qinf'],
        'LSAW_SOS':['Class', 'AUROC-NS', 'AUROC-LLK', 'AUROC-REC','AUROC-q1','AUROC-q2','AUROC-qinf','PRCISION','F1','RECALL', 'precision_q1', 'f1_q1','recall_q1','precision_q2', 'f1_q2','recall_q2','precision_qinf', 'f1_qinf','recall_qinf'],
        'LSA_MAF':['Class', 'AUROC-NS', 'AUROC-LLK', 'AUROC-REC','AUROC-q1','AUROC-q2','AUROC-qinf','PRCISION','F1','RECALL','precision_q1', 'f1_q1','recall_q1','precision_q2', 'f1_q2','recall_q2','precision_qinf', 'f1_qinf','recall_qinf'],

        'LSA_EN':['Class', 'AUROC-NS', 'AUROC-LLK', 'AUROC-REC','PRCISION','F1','RECALL'],
        'LSAW_EN':['Class', 'AUROC-NS', 'AUROC-LLK', 'AUROC-REC','PRCISION','F1','RECALL'],
        'LSA':['Class','AUROC'],
        'LSAD':['Class','AUROC'],
        'LSAW':['Class','AUROC'],
        'SOS':['Class','AUROC','PRCISION','F1','RECALL','precision_q1', 'f1_q1','recall_q1','precision_q2', 'f1_q2','recall_q2','precision_qinf', 'f1_qinf','recall_qinf'],

        # 'threshold':['Class', 'precision_den', 'f1_den', 'recall_den','acc_den',
        
        # # 'precision_q1','f1_q1','recall_q1',
        # # 'precision_q2','f1_q2','recall_q2', 
        # 'precision_qinf','f1_qinf','recall_qinf','acc_qinf','tn_n']
        }[self.name]

        # format
        table.float_format = '0.4'
        return table


    
    def compute_AUROC(self, log_step = 200, epoch_max = 20000):
        
        bs = self.batch_size
        auroc_dict = {
            'ns':[],
            'nllk':[],
            'rec':[],
            'q1':[],
            'q2':[],
            'qinf':[]}

        for cl_idx, cl in enumerate(self.dataset.test_classes):
            for epoch in range(0 , epoch_max+log_step, log_step):
                print(f"epoch:{epoch}")
                print(f"Testinng on {cl}")
                
                model_dir_epoch = join(self.checkpoints_dir,f'{cl}{self.name}_{self.train_strategy}_{epoch}.pkl')
                if epoch == epoch_max:
                    model_dir_epoch = join(self.checkpoints_dir,f'{cl}{self.name}_{self.train_strategy}.pkl')

                self.model.load_w(model_dir_epoch)
                print(f"Load Model from {model_dir_epoch}")
                
                self.model.eval()

                self.dataset.test(cl)
                data_num = self.dataset.length
                loader = DataLoader(self.dataset, batch_size = self.batch_size, shuffle = False)
                # density related

                sample_q1 = np.zeros(shape=(len(self.dataset),))
                sample_q2 = np.zeros(shape=(len(self.dataset),))
                sample_qinf = np.zeros(shape=(len(self.dataset),))
                sample_u = np.zeros(shape=(len(self.dataset), self.code_length))
                
                sample_llk = np.zeros(shape=(len(self.dataset),))
                sample_nrec = np.zeros(shape=(len(self.dataset),))
                # true label
                sample_y = np.zeros(shape=(len(self.dataset),))

                # TEST
                for i, (x, y) in tqdm(enumerate(loader), desc=f'Computing scores for {self.dataset}'):
                    
                    x = x.to(self.device)
                    with torch.no_grad():
                        if self.name in ['LSA_SOS','SOS','LSA_MAF','LSAD_SOS','LSAW_SOS']:
                            tot_loss, q1, q2, qinf, u = self._eval(x, average = False, quantile_flag= True)
                            # quantile 
                            sample_q1[i*bs:i*bs+bs] = q1
                            sample_q2[i*bs:i*bs+bs] = q2
                            sample_qinf[i*bs:i*bs+bs] = qinf
                            # source point 
                            sample_u[i*bs:i*bs+bs] = u
                        else:
                            tot_loss = self._eval(x, average = False)
                                
                    sample_y[i*bs:i*bs+bs] = y
                

                    if self.name in ['LSA','LSAD','LSAW','LSA_SOS','LSA_EN','LSA_MAF','LSAD_SOS','LSAW_SOS']:
                        sample_nrec[i*bs:i*bs+bs] = - self.loss.reconstruction_loss.cpu().numpy()
                            
                    if self.name in ['LSA_SOS','LSA_EN','EN','SOS','LSA_MAF','LSAD_SOS','LSAW_SOS']:    
                        sample_llk[i*bs:i*bs+bs] = - self.loss.autoregression_loss.cpu().numpy()
                    
                    sample_ns = novelty_score(sample_llk, sample_nrec)

                auroc_dict['ns'].append(roc_auc_score(sample_y, sample_ns))
                auroc_dict['nllk'].append(roc_auc_score(sample_y, sample_llk))
                auroc_dict['rec'].append(roc_auc_score(sample_y, sample_nrec))
                auroc_dict['q1'].append(roc_auc_score(sample_y,sample_q1))
                auroc_dict['q2'].append(roc_auc_score(sample_y, sample_q2))
                auroc_dict['qinf'].append(roc_auc_score(sample_y,sample_qinf))

            # write AUROC
                
            filename = f'auroc/{self.dataset.name}_c{cl}_{self.name}_{self.train_strategy}_auroc'
            outfile = open(filename,'wb')

            pickle.dump(auroc_dict,outfile)
            outfile.close()




    def plot_training_loss_auroc (self, log_step = 200):

        model_name = self.name
        train_strategy = self.train_strategy
        dataset_name = self.dataset.name

        for cl_idx, cl in enumerate(self.dataset.test_classes):
            train_history_path = join(self.checkpoints_dir,f'{cl}{model_name}_{train_strategy}_loss_history.npz')

            with np.load(train_history_path, allow_pickle = True) as data:

                loss_history = data['loss_history']
                train_loss = loss_history.item().get('train_loss')
                train_rec = loss_history.item().get('train_rec')
                train_nllk = loss_history.item().get('train_nllk')

                validation_loss =  loss_history.item().get('validation_loss')
                validation_rec = loss_history.item().get('validation_rec')
                validation_nllk =  loss_history.item().get('validation_nllk')

                best_validation_epoch = data['best_validation_epoch'] 
                best_validation_rec_epoch= data['best_validation_rec_epoch']


            print (f"Best validate loss Epoch:\t{best_validation_epoch}-loss\t{validation_loss[best_validation_epoch]}")
            print (f"Best_validate Reconstruction-Loss Epoch\t{best_validation_rec_epoch}-loss\t{validation_rec[best_validation_rec_epoch]}")

            x = range(0,len(train_loss),1)
            fig = plt.figure(0)

            if self.name in ['LSA_MAF','LSA_EN','LSA_SOS','LSAD_SOS','LSAW_SOS']:
                ax1 =plt.subplot(411)
                ax1.plot(x, train_loss, 'b',label = 'train_loss')
                ax1.plot(x, validation_loss, 'r',label = 'validation_loss')
                ax1.legend(loc= 'upper left')
                ax1.set_ylabel('loss')

                ax2 = plt.subplot(412)
                ax2.plot(x, train_rec, 'b',label = 'train_rec')
                ax2.plot(x, validation_rec, 'r',label = 'validation_rec')
                ax2.legend(loc= 'upper left')

                ax3 = plt.subplot(413)
                ax3.plot(x, train_nllk, 'b',label = 'train_nllk')
                ax3.plot(x, validation_nllk, 'r',label = 'validation_nllk')
                ax3.legend(loc= 'upper left')
            else:
                ax1 = plt.subplot()
                ax1.plot(x, train_loss, 'b',label = 'train_loss')
                ax1.plot(x, validation_loss, 'r',label = 'validation_loss')
                ax1.legend(loc= 'upper left')
                ax1.set_ylabel('loss')

            
            fig.suptitle(f"{dataset_name}_{cl}{model_name}_{train_strategy}_loss_history")

            auroc_file_name = f'auroc/{dataset_name}_c{cl}_{model_name}_{train_strategy}_auroc'

            pickle_in = open(auroc_file_name,"rb")
            auroc = pickle.load(pickle_in)

            print(auroc.keys())
            auroc_len = len(auroc['ns'])
            x = range(0, auroc_len*log_step, log_step)
            ax4 = ax1.twinx()  # this is the important function
            ax4.plot(x, auroc['nllk'], 'go-',label = 'nllk-auroc')
            ax4.plot(x, auroc['rec'],'ro-',label = 'rec-auroc')
            ax4.plot(x, auroc['ns'],'bo-',label = 'ns-auroc')
            
            ax4.set_ylabel('AUROC')
            ax4.legend(loc= 'lower right')
            # ax4.set_ylim((0.9,1))

            plt.savefig(f'distgraph/{dataset_name}_{cl}{model_name}_{train_strategy}_loss_history.png')
            plt.close(0)

            # auroc
            fig = plt.figure(1)
            # plt.axis([xmin, xmax, ymin, ymax])

            plt.plot(x, auroc['ns'], 'bo-',label = 'ns')
            
            if model_name in ['LSA_MAF','LSA_SOS','LSA_EN','LSAD_SOS','LSAW_SOS']:
                plt.plot(x, auroc['rec'], 'g--',label = 'rec')
                plt.plot(x, auroc['nllk'], 'r>-',label = 'nllk')

            if model_name in ['LSA_MAF','LSA_SOS','LSAD_SOS','LSAW_SOS']:
                  plt.plot(x, auroc['q1'], 'co-',label = 'q1')
                  plt.plot(x, auroc['q2'], 'm--',label = 'q2')
                  plt.plot(x, auroc['qinf'], 'y<-',label = 'qinf')

            plt.legend(loc='lower right')


            # plot_AUROC
            plt.savefig(f'distgraph/{dataset_name}_c{cl}_{model_name}_{train_strategy}_auroc.png')
            plt.close(1)






    def visualize_latent_vector(self, cl):
        from pandas.plotting import parallel_coordinates

    # Visualize feature maps
        self.get_path(cl)
        
        def get_latent_vector(train_strategy):
        
            if (self.test_checkpoint !=None):
                # select one epoch to test
                self.model_dir = join(self.checkpoints_dir,f'{cl}{self.name}_{train_strategy}_{self.test_checkpoint}.pkl')
                self.test_result_dir = join(self.checkpoints_dir,f'{cl}{self.name}_{train_strategy}_{self.test_checkpoint}_history_test')
            
            # Load the checkpoint 
            bs = self.batch_size   
            self.model.load_w(self.model_dir)
            print(f"Load Model from {self.model_dir}")

            self.model.eval()
            self.dataset.test(cl) 
            loader = DataLoader(self.dataset, batch_size = bs, shuffle = False)

            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output.detach()
                return hook

            self.model.encoder.register_forward_hook(get_activation('encoder'))
            sample_act = np.zeros(shape=(len(self.dataset), self.code_length))
                    
            # true label
            sample_y = np.zeros(shape=(len(self.dataset),))

            # compute feature map
            for i, (x, y) in tqdm(enumerate(loader), desc=f'Computing scores for {self.dataset}'):  
                x = x.to(self.device)
                # x = torch.randn(1, 1, 28, 28)
                self.model(x)
                act = activation['encoder'].squeeze()
                dimension_num = act.size(0)
                act = act.cpu().numpy()

                sample_y[i*bs:i*bs+bs] = y
                sample_act[i*bs:i*bs+bs] = act
            return sample_act, sample_y


        # Select features to include in the plot
        # plot_feat = range(dimension_num)

        # idx = np.arange(len(sample_y))
        # np.random.shuffle(idx)
        # # sample_act = sample_act[idx[0:200],0:10]
        # # sample_y = sample_y[idx[0:200]]

        # sample_act_1 = np.mean(sample_act[sample_y==1], axis =0)
        # sample_act_0 = np.mean(sample_act[sample_y==0], axis =0)
        # sample_y = [0,1]
        # sample_act = [sample_act_0,sample_act_1]


        # mat_data = np.mat(sample_act)
        # mat_data = mat_data.transpose()

        sample_act, sample_y = get_latent_vector('fix')
        sample_act2, sample_y2 = get_latent_vector('mul')
        sample_tr = ["fix"]*len(sample_y)
        sample_tr2 = ["mul"]*len(sample_y)
        
        
        data_dict = {}

        # for idx in range(mat_data.shape[0]):
        #     arr = np.array(mat_data[idx,:])
        #     lst = list(arr)
        #     lst = list(lst[0])
        #     data_dict[str(idx)] = lst
        # print(sample_act.shape)
        # print(sample_act2.shape)
        sample_act = np.append(sample_act.squeeze(), sample_act2.squeeze()).reshape(20000,64)
        # print(sample_act.shape)

        data_dict['activation'] =  sample_act[:,1]
        data_dict['label'] = np.append(sample_y.squeeze(), sample_y2.squeeze())
        data_dict['train_strategy'] = np.append(np.array(sample_tr).squeeze(),np.array(sample_tr2).squeeze())

        print(sum(sample_y))
        plt.figure(0)
        sns.violinplot(y='activation', x = 'label', data=data_dict, hue = 'train_strategy',split=False)
        
        plt.savefig("distgraph/1_feature.png")
        plt.close(0)
        # kernels = self.model.encoder.conv[0].conv1a.weight.cpu().detach().clone()
        # kernels = kernels - kernels.min()
        # kernels = kernels / kernels.max()
        # custom_viz(kernels, f'distgraph/{cl}{self.name}_{self.train_strategy}_conv1_weights.png', 4)


