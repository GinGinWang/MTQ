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

# from prettytable import PrettyTable
from tqdm import tqdm

from scipy.stats import norm
def _init_fn():
    np.random.seed(12)

class OneClassTestHelper(object):
    """
    Performs tests for one-class datasets (MNIST or CIFAR-10).
    """

    def __init__(
        self, dataset, 
        model, 
        score_normed , 
        lam , 
        checkpoints_dir, 
        result_file_path, 
        batch_size , 
        lr , 
        epochs, 
        before_log_epochs, 
        code_length,
        mulobj, 
        test_checkpoint, 
        log_step,
        device,
        fixed = False,
        pretrained= False,
        load_lsa = False,
        ):

        # type: (OneClassDataset, BaseModule, str, str) -> None
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
        torch.save(self.model.state_dict(), join(checkpoints_dir, f'{model.name}_start.pkl'))
        
        self.checkpoints_dir = checkpoints_dir
        self.log_step = log_step
        self.test_checkpoint = test_checkpoint
        self.result_file_path = result_file_path
        self.name = model.name
        self.batch_size = batch_size
        self.lam = lam

        # training-strategy
        if mulobj: 
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


        ## Set up loss function
        # encoder + decoder
        if self.name == 'LSA':
            self.loss = LSALoss(cpd_channels=100)
        # encoder + estimator+ decoder
        elif self.name == 'LSA_EN':
            self.loss = LSAENLoss(cpd_channels=100,lam=lam)
        
        elif self.name in ['LSA_SOS', 'LSA_MAF']:
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
        self.optimizer = optim.Adam(self.model.parameters(), lr= lr, weight_decay=1e-6)
        if self.fixed or self.pretrained:
            if (self.model.estimator !=None):
                self.est_optimizer = optim.Adam(self.model.estimator.parameters(),lr = lr, weight_decay = 1e-6)
            self.ae_optimizer  = optim.Adam(list(self.model.encoder.parameters())+list(self.model.decoder.parameters()),lr = lr, weight_decay = 1e-6)

        self.lr = lr
        self.train_epochs = epochs
        self.before_log_epochs = before_log_epochs
        
        
    def get_path(self):
        name    = self.name
        cl      = self.cl 
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

        
    def _eval_quantile(self, s, method_name='n_cdf'):
    #  from s~N(R^d) to u~(0,1)^d
    #  u_i = phi(s_i), where phi is the cdf of N(0,1)
    # Compute S 
    # only for SOS density estimator

        # if self.name =='LSA_SOS':
        #     _, _, s, _ = self.model(x)
        # elif self.name =='SOS':
        #     s, _ = self.model(x)
        # else:
        #      ValueError("Quantiles only computed for SOS Density Estimator")

        bs = s.shape[0]
        s_dim = s.shape[1]
        s_numpy = s.cpu().numpy()
        q1 = []
        q2 = []
        qinf = []
        if method_name=='n_cdf':
            for i in range(bs):
                # for every sample
                # cdf 
                u_s = norm.cdf(s_numpy[i,:]) ## Source point in the source uniform distribution
                # u = abs(np.ones((1,s_dim))*0.5-u_s)
                u = abs(0.5-u_s)
                if self.code_length>1:
                    uq_1 = np.linalg.norm(u,1)
                    uq_2 = np.linalg.norm(u)
                    uq_inf = np.linalg.norm(u,np.inf)
                else:
                    uq_1 = u
                    uq_2 = u
                    uq_inf = u

                # uq_inf = np.max(u)
                q1.append(-uq_1)
                q2.append(-uq_2)
                qinf.append(-uq_inf)
        else:
            ValueError("Unknown Mapping")

        return q1, q2, qinf, u_s
    
    
    
    def _eval(self, x, average = True, quantile_flag = False):

        if self.name in ['LSA']:
            # ok
            x_r = self.model(x)
            tot_loss = self.loss(x, x_r, average)

        elif self.name == 'LSA_EN':
            x_r, z, z_dist = self.model(x)
            tot_loss = self.loss(x, x_r, z, z_dist, average)

        elif self.name in ['LSA_SOS','LSA_MAF']:

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
      
            epoch_loss = 0
            epoch_recloss = 0
            epoch_nllk = 0
            
            bs      = self.batch_size

            self.dataset.train(cl)

            loader = DataLoader(self.dataset, batch_size = self.batch_size, shuffle= False, worker_init_fn = _init_fn, num_workers = 0)
            
            dataset_name = self.dataset.name
            epoch_size = self.dataset.length
            # pbar = tqdm(total=epoch_size)
            s_alpha = 0

            for batch_idx, (x , _) in enumerate(loader):

                # if batch_idx == 0:
                #     print(x.sum())

                x = x.to(self.device)
                self._eval(x)
                # self.optimizer.zero_grad()
                # backward average loss along batch
                if (self.mulobj):
                # Multi-objective Optimization
                    # g1: the gradient of reconstruction loss w.r.t the parameters of encoder
                    # g2: the gradient of auto-regression loss w.r.t the parameters of encoder
                    # Backward Total loss= Reconstruction loss + Auto-regression Loss
                    torch.autograd.backward(self.loss.autoregression_loss+self.loss.reconstruction_loss, self.model.parameters(),retain_graph =True)
                    #g1_list = g1 + g2
                    g1_list= [pi.grad.data.clone() for pi in list(self.model.encoder.parameters())]    
                    # Backward Auto-regression Loss, the gradients computed in the first backward are not cleared
                    # torch.autograd.backward(self.loss.autoregression_loss,list(self.model.encoder.parameters())+list(self.model.estimator.parameters()))
                    # torch.autograd.backward(self.loss.autoregression_loss, self.model.parameters())
                    torch.autograd.backward(self.loss.autoregression_loss, self.model.encoder.parameters())

                    
                    #g2_list = g1_list + g2
                    g2_list= [pi.grad.data.clone() for pi in list(self.model.encoder.parameters())]
                    
                    # the gradients w.r.t estimator are accumulated, div 2 to get the original one
                    # for p in self.model.estimator.parameters():
                    #     p.grad.data.div(2.0)

                    # compute alpha
                    top = 0
                    down = 0
                

                    # the formula (4) in Multi_Task Learning as Multi-Objective Function
                    i =0
                    for p in self.model.encoder.parameters():
                        g2   =  (g2_list[i]-g1_list[i])
                        g1   =  (g1_list[i]-g2)

                        g1 = g1.view(-1,)
                        g2 = g2.view(-1,)


                        top   =  top + torch.dot((g2-g1),g2).sum()

                        down  =  down+ torch.dot((g1-g2),(g1-g2)).sum()
                        i     =  i + 1

                    # print(top)
                    alpha = (top/down).item()
                    alpha = max(min(alpha,1),0)
                    # print(alpha)
                    
                    # compute new gradient of Shared Encoder by combined two gradients
                    i=0
        
                    s_alpha =s_alpha + alpha*x.shape[0]

                    for p in self.model.encoder.parameters():
                        newlrg2 = g2_list[i]-g1_list[i]
                        newlrg1 = 2*g1_list[i]-g2_list[i]
                        # compute the multi-gradient of the parameters in the encoder
                        p.grad.zero_()
                        p.grad.data = torch.mul(newlrg1,alpha)+torch.mul(newlrg2, 1-alpha)
                        i = i+1

                    self.optimizer.step()

                elif self.fixed:
                    if (not self.ae_finished):
                        self.loss.reconstruction_loss.backward()
                        self.ae_optimizer.step()
                    else:
                        self.loss.autoregression_loss.backward()
                        self.est_optimizer.step()

                elif self.pretrained:
                    if (not self.ae_finished):
                        self.loss.reconstruction_loss.backward()
                        self.ae_optimizer.step()
                    else:
                        self._eval(x).backward()
                        self.optimizer.step()
                else:
                    self.loss.total_loss.backward()
                    self.optimizer.step()
                     

                epoch_loss +=  self.loss.total_loss.item()*x.shape[0]

                if self.name in ['LSA_EN','LSA_SOS','LSA_MAF']:
                    # print(self.loss.reconstruction_loss)
                    # print(self.loss.autoregression_loss)
                    epoch_recloss += self.loss.reconstruction_loss.item()*x.shape[0]
                    epoch_nllk +=  self.loss.autoregression_loss.item()*x.shape[0]


                # pbar.update(x.size(0))
                # pbar.set_description('Train, Loss: {:.6f}'.format(epoch_loss / (pbar.n)))

                
                # images
                # plot_source_dist_by_dimensions(sample_u, sample_y, f"{self.train_result_dir}_{epoch}")

            # pbar.close()
            # if (epoch % 100 ==0) and (self.name in ['LSA_SOS','SOS']):
            #     self.train_validate(epoch,loader,bs)

            # print epoch result
            # if self.name in ['LSA_SOS']:
                
            #     print('{}Train Epoch-{}: {}\tLoss: {:.6f}\tRec: {:.6f}\tNllk: {:.6f}\tNllk1: {:.6f}\tNjob: {:.6f}'.format(self.name,
            #             self.dataset.normal_class, epoch, epoch_loss/epoch_size, epoch_recloss/epoch_size, epoch_nllk/epoch_size, epoch_nllk1/epoch_size, epoch_njob/epoch_size))


            if self.name in ['LSA_EN','LSA_SOS','LSA_MAF']:

                print('{}Train Epoch-{}: {}\tLoss: {:.6f}\tRec: {:.6f}\tNllk:{:.6f}\t'.format(self.name,
                        self.dataset.normal_class, epoch, epoch_loss/epoch_size, epoch_recloss/epoch_size, epoch_nllk/epoch_size))

            else:
                print('Train Epoch-{}: {}\tLoss:{:.6f}\t'.format(
                        self.dataset.normal_class, epoch, epoch_loss/epoch_size))

            if self.mulobj:
                print (f'Adaptive Alpha:{s_alpha/epoch_size}')

            return epoch_loss/epoch_size, epoch_recloss/epoch_size, epoch_nllk/epoch_size

    def validate(self, epoch, cl):

        prefix = 'Validation'
        self.model.eval()

        val_loss = 0
        val_nllk=0
        val_rec =0
        bs = self.batch_size

        self.dataset.val(cl)
        loader = DataLoader(self.dataset, self.batch_size, shuffle = False)

        epoch_size = self.dataset.length
        batch_size = len(loader)

        # if (epoch % 100 ==0) and (self.name in ['LSA_SOS','SOS']):
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
                # if (epoch % 100 ==0) and (self.name in ['LSA_SOS','SOS']):
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


                if self.name in ['LSA_EN','LSA_SOS','LSA_MAF']:
                    val_nllk += self.loss.autoregression_loss.sum().item()
                    val_rec += self.loss.reconstruction_loss.sum().item()
                    val_loss = val_nllk + val_rec
                else:
                     val_loss += self.loss.total_loss.sum().item()

                 


        # if (epoch % 100 == 0) and (self.name in ['LSA_SOS','SOS']):
        #     # save test-historty
        #     np.savez(f"{self.valid_result_dir}_{epoch}", 
        #             sample_y = sample_y, 
        #             sample_nrec = sample_nrec, 
        #             sample_llk = sample_llk,
        #             sample_qinf = sample_qinf,
        #             sample_u = sample_u)
            

                                    

        if self.name in ['LSA_EN','LSA_SOS','LSA_MAF']:
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
        
        for epoch in range(self.train_epochs):
            
            model_dir_epoch = join(self.checkpoints_dir,f'{cl}{self.name}_{self.train_strategy}_{epoch}.pkl')
            
            #train every epoch
            train_loss, train_rec, train_nllk= self.train_every_epoch(epoch,cl)      
            
            # validate every epoch
            validation_loss,validation_rec,validation_nllk = self.validate(epoch, cl)

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
                if (epoch>self.before_log_epochs):
                    torch.save(best_model.state_dict(), self.best_model_dir)
                print(f'Best_valid_epoch at :{epoch} with valid_loss:{best_validation_loss}' )

            if (epoch % self.log_step == 0 ) :
                    torch.save(self.model.state_dict(), model_dir_epoch)
            
            # early stop
            # if (epoch - best_train_epoch)> 50 and ((not self.fixed) or (self.fixed and self.ae_finished)):
            #     if (epoch-best_train_rec_epoch) > 50 and (epoch- best_train_nllk_epoch)>50:
            #         print (f"Break at Epoch:{epoch}")
            #         break

            if (self.fixed or self.pretrained) and(not self.ae_finished):

                if (validation_rec < best_validation_rec): 
                    best_validation_rec = validation_rec
                    best_validation_rec_epoch = epoch
                    best_rec_model = self.model
                    torch.save(best_rec_model.state_dict(), self.best_model_rec_dir)
                    print(f'Best_epoch at :{epoch} with rec_loss:{best_validation_loss}' )

                # if (best_train_rec_epoch ==epoch):
                #     best_rec_model = self.model
                #     torch.save(best_rec_model.state_dict(), f'{self.best_model_rec_dir}.pkl')
                #     print(f'Best_epoch at :{epoch} with rec_loss:{best_train_loss}' )
                
                if (epoch - best_validation_rec_epoch)> 50:
                    self.ae_finished = True  # autoencoder finished
                    state_rec = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),'optimizer': self.optimizer.state_dict()}
                    torch.save(state_rec, self.best_model_rec_detail_dir)
                    best_validation_epoch = epoch
                    best_validation_loss = float('+inf')
                    best_validation_rec = float('+inf')
                    best_validation_nllk = float('+inf')




            
        print("Training finish! Normal_class:>>>>>", cl)
        
        torch.save(self.model.state_dict(), self.model_dir)
        
        state = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
             'optimizer': self.optimizer.state_dict()}
        torch.save(state, self.model_detail_dir)
        
        np.savez(self.train_history_dir, loss_history= loss_history, best_validation_epoch = best_validation_epoch, best_validation_rec_epoch= best_train_rec_epoch)

    def test_one_class_classification(self, cl):
        
        # TEST FOR specific epoch
        if (self.test_checkpoint !=None):
            # select one epoch to test
            self.model_dir = join(self.checkpoints_dir,f'{cl}{self.name}_{self.train_strategy}_{self.test_checkpoint}.pkl')
            self.test_result_dir = join(self.checkpoints_dir,f'{cl}{self.name}_{self.train_strategy}_{self.test_checkpoint}_history_test')

        # Load the checkpoint 
        bs = self.batch_size   
        self.model.load_w(self.model_dir)
        print(f"Load Model from {self.model_dir}")

        self.model.eval()
        # normalizing coefficient of the Novelty Score (Eq.9 in LSA)
        min_llk, max_llk, min_rec, max_rec,min_q1,max_q1,min_q2,max_q2,min_qinf,max_qinf = self.compute_normalizing_coefficients(cl)
        # Test sets
        self.dataset.test(cl)
        
        loader = DataLoader(self.dataset, batch_size = bs, shuffle = False)

        # density related 
        sample_llk = np.zeros(shape=(len(self.dataset),))
        sample_nrec = np.zeros(shape=(len(self.dataset),))
        # true label
        sample_y = np.zeros(shape=(len(self.dataset),))
        # quantile related 
        sample_q1 = np.zeros(shape=(len(self.dataset),))
        sample_q2 = np.zeros(shape=(len(self.dataset),))
        sample_qinf = np.zeros(shape=(len(self.dataset),))
        # source point u (64)
        sample_u = np.zeros(shape=(len(self.dataset), self.code_length))
        
                
        # TEST
        for i, (x, y) in tqdm(enumerate(loader), desc=f'Computing scores for {self.dataset}'):  
            x = x.to(self.device)
            with torch.no_grad():
                if self.name in ['LSA_SOS','SOS','LSA_MAF']:
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
            # score larger-->normal data
            if self.name in ['LSA','LSA_SOS','LSA_EN','LSA_MAF']:
                sample_nrec[i*bs:i*bs+bs] = - self.loss.reconstruction_loss.cpu().numpy()
                
            if self.name in ['LSA_SOS','LSA_EN','EN','SOS','LSA_MAF']:    
                sample_llk[i*bs:i*bs+bs] = - self.loss.autoregression_loss.cpu().numpy()

        sample_llk = modify_inf(sample_llk)

        if self.score_normed:
            # Normalize scores
            sample_llk= normalize(sample_llk, min_llk, max_llk)
            sample_nrec = normalize(sample_nrec, min_rec, max_rec)
        
        sample_ns = novelty_score(sample_llk, sample_nrec)
        sample_ns = modify_inf(sample_ns)

        if self.name =='LSA_SOS':
            sample_ns_t = sample_llk # larger, normal
        else:
            sample_ns_t = sample_ns # larger, normal

        precision_den, f1_den, recall_den = compute_metric(self.name, sample_ns_t,sample_y)

        # # based on quantile-norm-inf
        if self.name in ['LSA_SOS','LSA_MAF']:
            this_class_metrics = [
            roc_auc_score(sample_y, sample_ns),
            roc_auc_score(sample_y, sample_llk),
            roc_auc_score(sample_y, sample_nrec),
            roc_auc_score(sample_y, sample_q1),
            roc_auc_score(sample_y, sample_q2),
            roc_auc_score(sample_y, sample_qinf),    #
            precision_den,
            f1_den,
            recall_den
            ]

        #     threshold_qinf = -pow((1-0.9),1/self.code_length)*0.5
        #     real_threshold = np.percentile(sample_qinf,real_nr*100)

        #     print(f"threshold_qinf:{threshold_qinf},vs{real_threshold}")
        #     print(np.max(sample_qinf))
        #     print(np.min(sample_qinf))
        #     y_hat_qinf = np.where((sample_qinf)>=(threshold_qinf), 1, 0)

            
        #     CM = confusion_matrix(sample_y==0, y_hat_qinf==0)
        #     TN = CM[0][0]
        #     FN = CM[1][0]

        #     if (FN + TN ==0):
        #         tn_n = 0
        #     else:
        #         tn_n = float(TN)/(float(FN+TN))
            
        #     print(f"Quantile-based, Predicted Novelty_Num: {sum(y_hat_qinf==0)} in {len(y_hat_qinf)} samples")
        #     ####################################################
        #     precision_qinf, recall_qinf, f1_qinf, _ = precision_recall_fscore_support((sample_y==0),(y_hat_qinf==0), average="binary")
        #     acc_qinf = accuracy_score((sample_y==0),(y_hat_qinf==0))
        #     # plot_source_dist_by_dimensions(sample_u, sample_y, self.test_result_dir) 
        # add rows
        
        # every row
        # that_class_metrics = [
        # precision_den,
        # f1_den,
        # recall_den,
        # acc_den,
        # precision_q1,
        # f1_q1,
        # recall_q1,
        # precision_q2,
        # f1_q2,
        # recall_q2,
        # precision_qinf,
        # f1_qinf,
        # recall_qinf,
        # acc_qinf,
        # tn_n
        # ]

        # add rows
        # threshold_table.add_row([cl_idx] + that_class_metrics)
        # another_all_metrics.append(that_class_metrics)

        elif self.name in ['LSA_EN']:
         # every row
            this_class_metrics = [
            roc_auc_score(sample_y, sample_ns),
            roc_auc_score(sample_y, sample_llk),
            roc_auc_score(sample_y, sample_nrec),
            precision_den,
            f1_den,
            recall_den
            ]
        elif self.name in ['LSA']:
        # every row
            this_class_metrics = [
            roc_auc_score(sample_y, sample_ns)
            ]

        return this_class_metrics

    def train_classification(self):
        bs =self.batch_size
        # Start iteration over classes
        for cl_idx, cl in enumerate(self.dataset.test_classes):
            self.cl = cl
            self.get_path()
            # initialization
            if self.load_lsa:
                self.model.load_w(f'/home/jj27wang/novelty-detection/NovelDect_SoS/SoSLSA/checkpoints/{self.dataset.name}/{cl}LSA_1.pkl')
                self.ae_finished = True
            else:
                self.model.load_w(join(self.checkpoints_dir, f'{self.model.name}_start.pkl'))
            
            self.train_one_class_classification(cl)

    def test_classification(self):
        all_metrics = []
        self.style = 'threshold'
        # threshold_table = self.empty_table
        another_all_metrics = []

        bs =self.batch_size
        
        # Start iteration over classes
        for cl_idx, cl in enumerate(self.dataset.test_classes):
            self.cl = cl
            self.get_path()
            one_class_metric= self.test_one_class_classification(cl)
            # auroc_table.add_row([cl_idx] + this_class_metrics)

            all_metrics.append(one_class_metric)

        all_metrics = np.array(all_metrics)
        avg_metrics = np.mean(all_metrics, axis=0)

        # another_all_metrics =np.array(another_all_metrics)
        # another_avg_metrics = np.mean(another_all_metrics, axis=0)


        # auroc_table.add_row(['avg'] + list(avg_metrics))
        # print(auroc_table)

        # threshold_table.add_row(['avg'] + list(another_avg_metrics))
        # print(threshold_table)
        
        # Save table
        # with open(self.result_file_path, mode='w') as f:
        #     f.write(str(auroc_table))
            # f.write(str(threshold_table))
        print (avg_metrics)



     
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

                if self.name in ['LSA_SOS','SOS']:
                    tot_loss, q1,q2,qinf,_ = self._eval(x, quantile_flag= True)
                    sample_q1[i*bs:i*bs+bs] = q1
                    sample_q2[i*bs:i*bs+bs] = q2
                    sample_qinf[i*bs:i*bs+bs] = qinf
                else:
                    tot_loss = self._eval( x, average = False)

            # score larger-->normal data
            if self.name in ['LSA','LSA_SOS','LSA_EN']:
                sample_nrec[i*bs:i*bs+bs] = - self.loss.reconstruction_loss.cpu().numpy()
        
            if self.name in ['LSA_SOS','LSA_EN','EN','SOS']:    
                sample_llk[i*bs:i*bs+bs] = - self.loss.autoregression_loss.cpu().numpy()

            
            
            sample_llk = modify_inf(sample_llk)

        return sample_llk.min(), sample_llk.max(), sample_nrec.min(), sample_nrec.max(),sample_q1.min(),sample_q1.max(),sample_q2.min(),sample_q2.max(),sample_qinf.min(), sample_qinf.max()

    # @property
    def empty_table(self):
        # type: () -> PrettyTable
        """
        Sets up a nice ascii-art table to hold results.
        This table is suitable for the one-class setting.
        :return: table to be filled with auroc metrics.
        """
        table = PrettyTable()
        style = self.style
        table.field_names = {

        'auroc':['Class', 'AUROC-NS', 'AUROC-LLK', 'AUROC-REC','AUROC-q1','AUROC-q2','AUROC-qinf','PRCISION','F1','RECALL'],


        'threshold':['Class', 'precision_den', 'f1_den', 'recall_den','acc_den',
        
        # 'precision_q1','f1_q1','recall_q1',
        # 'precision_q2','f1_q2','recall_q2', 
        'precision_qinf','f1_qinf','recall_qinf','acc_qinf','tn_n']
        }[style]

        # format
        table.float_format = '0.4'
        return table


    def compute_AUROC(self, epoch_min = 0, epoch_max = 3000, epoch_step =50, cl = 0):
        bs = self.batch_size
        import pickle
        auroc_dict = {'ns':[],
            'nllk':[],
            'rec':[],
            'q1':[],
            'q2':[],
            'qinf':[]}

        
        for epoch in range(0 , 3000, 50):
            print(f"epoch:{epoch}")
            print(f"Testinng on {cl}")
            
            model_dir_epoch = join(self.checkpoints_dir,f'{cl}{self.name}_{self.train_strategy}_{epoch}.pkl')

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
                    if self.name in ['LSA_SOS','SOS','LSA_MAF']:
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
            

                if self.name in ['LSA','LSA_SOS','LSA_EN','LSA_MAF']:
                    sample_nrec[i*bs:i*bs+bs] = - self.loss.reconstruction_loss.cpu().numpy()
                        
                if self.name in ['LSA_SOS','LSA_EN','EN','SOS','LSA_MAF']:    
                    sample_llk[i*bs:i*bs+bs] = - self.loss.autoregression_loss.cpu().numpy()
                
                sample_ns = novelty_score(sample_llk, sample_nrec)

            auroc_dict['ns'].append(roc_auc_score(sample_y, sample_ns))
            auroc_dict['nllk'].append(roc_auc_score(sample_y, sample_llk))
            auroc_dict['rec'].append(roc_auc_score(sample_y, sample_nrec))
            auroc_dict['q1'].append(roc_auc_score(sample_y,sample_q1))
            auroc_dict['q2'].append(roc_auc_score(sample_y, sample_q2))
            auroc_dict['qinf'].append(roc_auc_score(sample_y,sample_qinf))




            # AUROC
            
        filename = f'c{cl}_{self.name}_{self.train_strategy}_auroc'
        outfile = open(filename,'wb')

        pickle.dump(auroc_dict,outfile)
        outfile.close()


    def add_two_pretrained_model(self):
        ####load model1
        dataset_name = self.dataset.name
        lam = self.lam
        cl = self.cl 

        model_dict = self.model.encoder.state_dict()
        pretrained_model1 = LSA_MNIST(input_shape=self.dataset.shape, code_length=64, num_blocks=1, est_name= 'SOS',hidden_size= 2048).cuda()
        pretrained_model1.load_w(f'checkpoints/{self.dataset.name}/b1h2048c64/{cl}LSA_SOS_mul_1000.pkl')
        print("Load the Pretrained LSA")

        pretrained_dict1= pretrained_model1.state_dict()
        pretrained_dict1 = {k: v for k, v in pretrained_dict1.items() if k in model_dict} 

        model_dict.update(pretrained_dict1) 
        self.model.load_state_dict(model_dict,strict = False)

        ######load model2
        model_dict = self.model.decoder.state_dict()

        pretrained_model2 = LSA_MNIST(input_shape=self.dataset.shape, code_length=64, num_blocks=1, est_name = 'SOS',hidden_size= 2048).cuda()
        print("Load the Pretrained LSA_ET")
        
        pretrained_model2.load_w(f'checkpoints/{self.dataset.name}/b1h2048c64/{cl}LSA_SOS_mul_1000.pkl')

        pretrained_dict2= pretrained_model2.state_dict()
        pretrained_dict2 = {k: v for k, v in pretrained_dict2.items() if k in model_dict}

        # update & load
        model_dict.update(pretrained_dict2) 
        self.model.load_state_dict(model_dict,strict= False)
        


