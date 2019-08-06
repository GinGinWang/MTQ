from os.path import join
from typing import Tuple
import torch.optim as optim
import numpy as np
import torch
from prettytable import PrettyTable
from torch.utils.data import DataLoader

from tqdm import tqdm

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



from models import *
import math

from scipy.stats import norm

class OneClassTestHelper(object):
    """
    Performs tests for one-class datasets (MNIST or CIFAR-10).
    """

    def __init__(
        self, dataset, 
        model, 
        score_normed , 
        novel_ratio, 
        lam , 
        checkpoints_dir, 
        result_file_path, 
        batch_size , 
        trainflag, 
        testflag, 
        lr , 
        epochs, 
        before_log_epochs, 
        noise, 
        noise2, 
        noise3,
        noise4, 
        code_length,
        mulobj, 
        test_checkpoint, 
        epoch_start,
        device,
        fixed = False
        ):

        # type: (OneClassDataset, BaseModule, str, str) -> None
        """
        Class constructor.

        :param dataset: dataset class.
        :param model: py-torch model to evaluate.
        :param score_normed: 1 normalized the novelty score with valid set, 0: not normalized
        :param novel_ratio: novel_ratio in test sets
        :param checkpoints_dir: directory holding checkpoints for the model.
        :param result_file_path: text file where to save results.
        """

        self.dataset = dataset
        self.model = model
        self.device = device

        # use the same initialization for all models
        torch.save(self.model.state_dict(), join(checkpoints_dir, f'{model.name}_start.pkl'))
        
        self.checkpoints_dir = checkpoints_dir
        self.test_checkpoint = test_checkpoint
        self.result_file_path = result_file_path
        self.name = model.name
        self.batch_size = batch_size
        self.lam = lam

        # test for robutstness
        self.noise = noise
        self.noise2 = noise2
        self.noise3 = noise3
        self.noise4 = noise4
        
        # control novel ratio in test sets.
        self.novel_ratio = novel_ratio

        self.trainflag = trainflag # whether need train
        self.testflag = testflag # whether test

        self.mulobj = mulobj# whether use mul-gradient
        self.score_normed = score_normed # normalized novelty score
        
        self.style = 'auroc' # table-style
        self.code_length = code_length

        self.ae_finished = False
        self.fixed = fixed 

        ## Set up loss function
        # encoder + decoder
        if self.name == 'LSA':
            self.loss = LSALoss(cpd_channels=100)
        # encoder + estimator+ decoder
        elif self.name == 'LSA_EN':
            self.loss = LSAENLoss(cpd_channels=100,lam=lam)
        
        elif self.name == 'LSA_SOS':
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
        if trainflag:
            self.optimizer = optim.Adam(self.model.parameters(), lr= lr, weight_decay=1e-6)
            self.est_optimizer = optim.Adam(self.model.estimator.parameters(),lr = lr, weight_decay = 1e-6)
            self.ae_optimizer  = optim.Adam(list(self.model.encoder.parameters())+list(self.model.decoder.parameters()),lr = lr, weight_decay = 1e-6)

            self.epoch_start = epoch_start
            self.lr = lr
            self.train_epochs = epochs
            self.before_log_epochs = before_log_epochs
        
        
    def get_path(self):
        name    = self.name
        cl      = self.cl 
        checkpoints_dir = self.checkpoints_dir
        test_checkpoint = self.test_checkpoint 
        lam     = self.lam 
        
        if self.mulobj: 
            nameflag = 'mul'
        else:
            nameflag= f'{self.lam}'

        self.model_dir = join(checkpoints_dir,f'{cl}{name}_{nameflag}.pkl')
        self.best_model_dir = join(checkpoints_dir,f'{cl}{name}_{nameflag}_b')
        self.best_model_rec_dir = join(checkpoints_dir,f'{cl}{name}_{nameflag}_rec')
        
        self.train_result_dir = join(checkpoints_dir,f'{cl}{name}_{nameflag}_history_train')
        self.valid_result_dir = join(checkpoints_dir,f'{cl}{name}_{nameflag}_history_valid')
        self.test_result_dir = join(checkpoints_dir,f'{cl}{name}_{nameflag}_history_test')

        if (not (self.test_checkpoint ==None)) and (not self.trainflag):
                # select one epoch to test
            self.model_dir = join(checkpoints_dir,f'{cl}{self.name}_{nameflag}_{test_checkpoint}.pkl')
            self.test_result_dir = join(checkpoints_dir,f'{cl}{name}_{nameflag}_{test_checkpoint}_history_test')

        
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
                u = abs(np.ones((1,s_dim))*0.5-u_s)
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
            tot_loss = self.loss(x, x_r, z, z_dist,average)

        elif self.name == 'LSA_SOS':
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
            # epoch_nllk1 = 0
            # epoch_njob =0


            noise   = self.noise
            noise2  = self.noise2
            noise3  =self.noise3
            noise4  = self.noise4
            bs      = self.batch_size

            self.dataset.train(cl, noise_ratio = noise, noise2_ratio = noise2, noise3_ratio = noise3, noise4_ratio = noise4)
            loader = DataLoader(self.dataset, batch_size = self.batch_size, shuffle= True)
            
            dataset_name = self.dataset.name
            epoch_size = self.dataset.length
            pbar = tqdm(total=epoch_size)
            s_alpha = 0
            self.model.train()
          
            for batch_idx, (x , _) in enumerate(loader):
                
                x = x.to(self.device)
                self.optimizer.zero_grad()
                # backward average loss along batch
                if (self.mulobj):
                    if self.ae_finished or (self.fixed ==False):
                    # Multi-objective Optimization
                        # g1: the gradient of reconstruction loss w.r.t the parameters of encoder
                        # g2: the gradient of auto-regression loss w.r.t the parameters of encoder
                        self._eval(x)
                        # Backward Total loss= Reconstruction loss + Auto-regression Loss
                        torch.autograd.backward(self.loss.autoregression_loss+self.loss.reconstruction_loss, self.model.parameters(),retain_graph =True)
                        #g1_list = g1 + g2
                        g1_list= [pi.grad.data.clone() for pi in list(self.model.encoder.parameters())]    
                        # Backward Auto-regression Loss, the gradients computed in the first backward are not cleared
                        torch.autograd.backward(self.loss.autoregression_loss,list(self.model.encoder.parameters())+list(self.model.estimator.parameters()))
                        # torch.autograd.backward(self.loss.autoregression_loss, self.model.parameters())
                        
                        #g2_list = g1_list + g2
                        g2_list= [pi.grad.data.clone() for pi in list(self.model.encoder.parameters())]
                        
                        # the gradients w.r.t estimator are accumulated, div 2 to get the original one
                        for p in self.model.estimator.parameters():
                            p.grad.data.div(2.0)

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

                            down  =  down+ torch.pow((g1-g2),2).sum()
                            i     =  i + 1

                        if down ==0:
                            alpha = 0.5
                        else:
                            # print(top)
                            alpha = (top/down).item()
                            alpha = max(min(alpha,1),0)
                            
                        # compute new gradient of Shared Encoder by combined two gradients
                        i=0
            
                        s_alpha =s_alpha + alpha*x.shape[0]

                        for p in self.model.encoder.parameters():
                            newlrg2 = g2_list[i]-g1_list[i]
                            newlrg1 = 2*g1_list[i]-g2_list[i]
                            # compute the multi-gradient of the parameters in the encoder
                            # p.grad.zero_()
                            p.grad.data = torch.mul(newlrg1,alpha)+torch.mul(newlrg1, 1-alpha)
                            i = i+1

                        self.optimizer.step()

                    else:
                        self._eval(x)
                        self.loss.reconstruction_loss.backward()
                        self.ae_optimizer.step()
                
                elif self.fixed:
                    self._eval(x)
                    
                    if (not self.ae_finished):
                        self.loss.reconstruction_loss.backward()
                        self.ae_optimizer.step()
                    else:
                        self.loss.autoregression_loss.backward()
                        self.est_optimizer.step()
                
                else:    
                    self._eval(x).backward()
                    self.optimizer.step()


                epoch_loss +=  self.loss.total_loss.item()*x.shape[0]

                if self.name in ['LSA_EN','LSA_SOS']:
                    epoch_recloss += self.loss.reconstruction_loss.item()*x.shape[0]
                    epoch_nllk +=  self.loss.autoregression_loss.item()*x.shape[0]

                    # if self.name in ['LSA_SOS']:
                    #     epoch_nllk1 +=  self.loss.nlog_probs.item()*x.shape[0]
                    #     epoch_njob+= self.loss.nagtive_log_jacob.item()*x.shape[0]


                pbar.update(x.size(0))
                pbar.set_description('Train, Loss: {:.6f}'.format(epoch_loss / (pbar.n)))

                
                # images
                # plot_source_dist_by_dimensions(sample_u, sample_y, f"{self.train_result_dir}_{epoch}")

            pbar.close()
            # if (epoch % 100 ==0) and (self.name in ['LSA_SOS','SOS']):
            #     self.train_validate(epoch,loader,bs)

            # print epoch result
            # if self.name in ['LSA_SOS']:
                
            #     print('{}Train Epoch-{}: {}\tLoss: {:.6f}\tRec: {:.6f}\tNllk: {:.6f}\tNllk1: {:.6f}\tNjob: {:.6f}'.format(self.name,
            #             self.dataset.normal_class, epoch, epoch_loss/epoch_size, epoch_recloss/epoch_size, epoch_nllk/epoch_size, epoch_nllk1/epoch_size, epoch_njob/epoch_size))


            if self.name in ['LSA_EN','LSA_SOS']:

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


                if self.name in ['LSA_EN','LSA_SOS']:
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
            

                                    

        if self.name in ['LSA_EN','LSA_SOS']:
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
        best_validation_rec_epoch = 0
       
        best_validation_loss = float('+inf')
        best_validation_rec = float('+inf')
        best_validation_nllk = float('+inf')

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

            epoch_new = epoch +int(self.epoch_start) 

            if self.mulobj: 
                nameflag = 'mul'
            else:
                nameflag= f'{self.lam}'

            model_dir_epoch = join(self.checkpoints_dir,f'{cl}{self.name}_{nameflag}_{epoch_new}.pkl')
            
            train_loss, train_rec, train_nllk= self.train_every_epoch(epoch,cl)      
            # validate
            validation_loss,validation_rec,validation_nllk = self.validate(epoch, cl)

            loss_history['train_loss'].append(train_loss)
            loss_history['train_rec'].append(train_rec)
            loss_history['train_nllk'].append(train_nllk)

            loss_history['validation_loss'].append(validation_loss)
            loss_history['validation_rec'].append(validation_rec)
            loss_history['validation_nllk'].append(validation_nllk)

            old_validation_loss = validation_loss

               
            if (validation_loss < best_validation_loss): 
                best_validation_loss = validation_loss
                best_validation_epoch = epoch
                best_model = self.model 
                if (epoch>self.before_log_epochs):
                    torch.save(best_model.state_dict(), f'{self.best_model_dir}.pkl')
                print(f'Best_epoch at :{epoch} with valid_loss:{best_validation_loss}' )


            if (epoch % 50 == 0 ) :
                    torch.save(self.model.state_dict(), model_dir_epoch)
                    # np.save(self.result_dir,history)
                    # save sample data
            # if (epoch - best_validation_epoch)> 30 and (self.name =='SOS'):
            #     print (f"Break at Epoch:{epoch}")
            #     break

            if self.fixed and (not self.ae_finished):
                if (validation_rec < best_validation_rec): 
                    best_validation_rec = validation_rec
                    best_validation_rec_epoch = epoch
                    best_rec_model = self.model
                    torch.save(best_rec_model.state_dict(), f'{self.best_model_rec_dir}.pkl')
                    print(f'Best_epoch at :{epoch} with rec_loss:{best_validation_loss}' )
                
                if (epoch - best_validation_rec_epoch)> 30:
                    self.ae_finished = True  # autoencoder finished
                    # self.model.load_w(join(self.checkpoints_dir, f'{self.best_model_rec_dir}_{epoch}.pkl'))


            
        print("Training finish! Normal_class:>>>>>", cl)
        
        torch.save(self.model.state_dict(), self.model_dir)
        
        np.savez(f'{self.checkpoints_dir}/{cl}_loss_history', loss_history= loss_history, best_validation_epoch = best_validation_epoch, best_validation_rec_epoch= best_validation_rec_epoch)

    def test_one_class_classification(self):
        # type: () -> None
        """
        Actually performs tests.
        """
        # Prepare a table to show results
        # Set up container for metrics from all classes
        auroc_table = self.empty_table
        all_metrics = []
        self.style = 'threshold'
        threshold_table = self.empty_table
        another_all_metrics = []


        bs =self.batch_size
        
        # Start iteration over classes
        for cl_idx, cl in enumerate(self.dataset.test_classes):
            self.cl = cl
            self.get_path()

            print(f"Testinng on {cl}")

            if self.trainflag:
                # initialization
                if self.epoch_start ==0:
                    # self.model.apply(weights_init)                
                    self.model.load_w(join(self.checkpoints_dir, f'{self.model.name}_start.pkl'))

                else:
                    self.model.load_w(self.model_dir)

                self.train_one_class_classification(cl)
            
            if self.testflag:
                # Load the checkpoint    
                self.model.load_w(self.model_dir)
                print(f"Load Model from {self.model_dir}")
                
                self.model.eval()
                # normalizing coefficient of the Novelty Score (Eq.9 in LSA)
                min_llk, max_llk, min_rec, max_rec,min_q1,max_q1,min_q2,max_q2,min_qinf,max_qinf = self.compute_normalizing_coefficients(cl)
                
                # Test sets
                
                self.dataset.test(cl, self.novel_ratio)
                data_num = self.dataset.length

                loader = DataLoader(self.dataset, batch_size = bs, shuffle = False)

                # density related 
                sample_llk = np.zeros(shape=(len(self.dataset),))
                sample_nrec = np.zeros(shape=(len(self.dataset),))
                # true label
                sample_y = np.zeros(shape=(len(self.dataset),))
                
                if self.name in ['LSA_SOS', 'SOS']:
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
                        if self.name in ['LSA_SOS','SOS']:
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
                    if self.name in ['LSA','LSA_SOS','LSA_EN']:
                        sample_nrec[i*bs:i*bs+bs] = - self.loss.reconstruction_loss.cpu().numpy()
                        
                    if self.name in ['LSA_SOS','LSA_EN','EN','SOS']:    
                        sample_llk[i*bs:i*bs+bs] = - self.loss.autoregression_loss.cpu().numpy()
                    


                    


                # +inf,-inf,nan
                sample_llk = modify_inf(sample_llk)

                llk1= np.dot(sample_llk,sample_y).sum()
                llk2 = sample_llk.sum()-llk1

                # llk1 should be larger than llk2
                llk1 =llk1/np.sum(sample_y)
                # average llk for normal examples
                llk2 =llk2/(data_num-np.sum(sample_y)) 
                # average llk for novel examples

                print(f'min_llk:{min_llk},max_llk:{max_llk}'
                        )
                print(f'min_rec:{min_rec},max_rec:{max_rec}')

                # Normalize scores
                sample_llk_n = normalize(sample_llk, min_llk, max_llk)
                sample_nrec_n = normalize(sample_nrec, min_rec, max_rec)
                
                # Compute the normalized novelty score
                if self.score_normed:
                    sample_nrec = sample_nrec_n
                    sample_llk = sample_llk_n    
                
                sample_ns = novelty_score(sample_llk, sample_nrec)
                sample_ns = modify_inf(sample_ns)

                # Compute precision, recall, f1_score based on threshold
                # if we know a/100 is the percentile of novelty samples in testset
                if self.name =='LSA_SOS':
                    sample_ns_t = sample_llk # larger, normal
                else:
                    sample_ns_t = sample_ns # larger, normal


                # y = 1 normal, y = 0 novel
                real_nr= float(sum(sample_y==0)/len(sample_y))            
                print(f"Real Novelty_Num: {sum(sample_y == 0)} in {len(sample_y)} samples, Novel Ratio= {real_nr}")


                #based on density(sort first)
                threshold1 = np.percentile(sample_ns_t, real_nr*100)
                print(f"threshold1:{threshold1}")

                y_hat1 = np.where(sample_ns_t >= threshold1, 1, 0)
                print(f"Density-based, Predicted Novelty_Num: {sum(y_hat1==0)} in {len(y_hat1)} samples")
                # wrong_predict1 = np.where(sample_y!= y_hat1)
                # print(f"Wrongly Predict on {len(wrong_predict1)}")                
                # ####################################################
                precision_den, recall_den, f1_den, _ =  precision_recall_fscore_support((sample_y==0),(y_hat1==0), average= "binary")
                acc_den = accuracy_score((sample_y==0),(y_hat1==0))

                ###################################################

                # based on quantile-norm-inf
                threshold_qinf = -pow((1-0.9),1/self.code_length)*0.5
                real_threshold = np.percentile(sample_qinf,real_nr*100)

                print(f"threshold_qinf:{threshold_qinf},vs{real_threshold}")
                print(np.max(sample_qinf))
                print(np.min(sample_qinf))
                y_hat_qinf = np.where((sample_qinf)>=(threshold_qinf), 1, 0)

                
                CM = confusion_matrix(sample_y==0, y_hat_qinf==0)
                TN = CM[0][0]
                FN = CM[1][0]

                if (FN + TN ==0):
                    tn_n = 0
                else:
                    tn_n = float(TN)/(float(FN+TN))
                
                print(f"Quantile-based, Predicted Novelty_Num: {sum(y_hat_qinf==0)} in {len(y_hat_qinf)} samples")
                ####################################################
                precision_qinf, recall_qinf, f1_qinf, _ = precision_recall_fscore_support((sample_y==0),(y_hat_qinf==0), average="binary")
                acc_qinf = accuracy_score((sample_y==0),(y_hat_qinf==0))

                ####################################################


                # save test-historty
                np.savez(self.test_result_dir, 
                    sample_y = sample_y, 
                    y_hat = y_hat_qinf,
                    sample_nrec = sample_nrec, 
                    sample_llk = sample_llk,
                    sample_qinf = sample_qinf,
                    sample_u = sample_u)
                # images
                # plot_source_dist_by_dimensions(sample_u, sample_y, self.test_result_dir) 
                
                #auroc-table
                # every row
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

                # add rows
                auroc_table.add_row([cl_idx] + this_class_metrics)
                all_metrics.append(this_class_metrics)

                # every row
                that_class_metrics = [
                precision_den,
                f1_den,
                recall_den,
                acc_den,
                # precision_q1,
                # f1_q1,
                # recall_q1,
                # precision_q2,
                # f1_q2,
                # recall_q2,
                precision_qinf,
                f1_qinf,
                recall_qinf,
                acc_qinf,
                tn_n
                ]

                # add rows
                threshold_table.add_row([cl_idx] + that_class_metrics)
                another_all_metrics.append(that_class_metrics)

            else:
                ValueError("Train or Test, at least one mode, --NoTrain, --NoTest")

                # Compute average AUROC and print table
        if self.testflag:
            all_metrics = np.array(all_metrics)
            avg_metrics = np.mean(all_metrics, axis=0)

            another_all_metrics =np.array(another_all_metrics)
            another_avg_metrics = np.mean(another_all_metrics, axis=0)

            auroc_table.add_row(['avg'] + list(avg_metrics))
            print(auroc_table)

            threshold_table.add_row(['avg'] + list(another_avg_metrics))
            print(threshold_table)
            
            # Save table
            with open(self.result_file_path, mode='w') as f:
                f.write(str(auroc_table))
                f.write(str(threshold_table))


     
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

    @property
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



