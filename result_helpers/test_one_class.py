from os.path import join
from typing import Tuple
import torch.optim as optim
import numpy as np
import torch
from prettytable import PrettyTable
from torch.utils.data import DataLoader

from tqdm import tqdm
from datasets.utils import set_random_seed

from datasets.base import OneClassDataset
from models.base import BaseModule
from models.loss_functions import * 
from datasets.utils import novelty_score
from datasets.utils import normalize

from result_helpers.utils import *

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from models import *
import math

# for cdf 
from scipy.stats import norm

# from result_helpers import metric_method as mm



class OneClassTestHelper(object):
    """
    Performs tests for one-class datasets (MNIST or CIFAR-10).
    """

    def __init__(self, dataset, model, score_normed, novel_ratio, lam, checkpoints_dir, output_file, device,batch_size, trainflag, lr, epochs, before_log_epochs, combined, pretrained, add,from_pretrained = False, fixed= False, pretrained_model = 'LSA',mulobj=False, quantile_flag = False):
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
        # save initialization
        torch.save(self.model.state_dict(), join(checkpoints_dir, f'{model.name}_start.pkl'))
        self.combined = combined
        self.checkpoints_dir = checkpoints_dir
        self.output_file = output_file
        self.device = device
        self.name = model.name
        self.batch_size = batch_size
        self.lam = lam

        # control novel ratio in test sets.
        self.novel_ratio = novel_ratio

        self.trainflag = trainflag # whether need train
        self.quantile_flag = quantile_flag # whether use quantile
        self.add = add # whether directly use two pretrained model
        self.mulobj= mulobj# whether use mul-gradient
        self.score_normed = score_normed # normalized novelty score
        

        # Set up loss function
        # encoder + decoder
        if self.name == 'LSA':
            self.loss =LSALoss(cpd_channels=100)
        # encoder + estimator+ decoder
        elif self.name == 'LSA_EN':
            self.loss = LSAENLoss(cpd_channels=100,lam=lam)
        elif self.name == 'LSA_SOS':
            self.loss =LSASOSLoss(lam)
        elif self.name == 'LSA_MAF':
            self.loss =LSAMAFLoss(lam)
        elif self.name == 'LSA_QT':
            self.loss = LSAQTLoss(lam)
        
        # encoder + estimator
        elif self.name == 'LSA_ET_EN':
            self.loss = LSAETENLoss(cpd_channels=100)
        elif self.name == 'LSA_ET_SOS':
            self.loss = LSAETSOSLoss()
        elif self.name == 'LSA_ET_MAF':
            self.loss =LSAETMAFLoss()
        elif self.name == 'LSA_ET_QT':
            self.loss =LSAETQTLoss() 
        
        else:
            ValueError("Wrong Model Name")
        
        print (f"Testing on {self.name}")

        # initialize dir
        self.model_dir = None
        self.best_model_dir = None
        self.result_dir = None

        # Related to training
        if trainflag:
            if fixed:
                # only train estimator
                self.optimizer = optim.Adam(self.model.estimator.parameters(),lr= lr, weight_decay = 1e-6)
            else:
                self.optimizer = optim.Adam(self.model.parameters(), lr= lr, weight_decay=1e-6)

            self.pretrained = pretrained
            self.from_pretrained = from_pretrained
            self.pretrained_model = pretrained_model
            self.lr = lr
            self.train_epoch = epochs
            self.before_log_epochs = before_log_epochs
        
        
    def get_path(self):
        name    = self.name
        cl      = self.cl 
        checkpoints_dir = self.checkpoints_dir 
        lam     = self.lam 
        
        if self.mulobj:
            self.model_dir = join(checkpoints_dir,f'{cl}{name}_mul.pkl')
            self.best_model_dir = join(checkpoints_dir,f'{cl}{name}_mul_b.pkl')
            self.result_dir = join(checkpoints_dir,f'{cl}{name}_mul_history.npy')
        else:
            self.model_dir = join(checkpoints_dir,f'{cl}{name}_{lam}.pkl')
            self.best_model_dir = join(checkpoints_dir,f'{cl}{name}_{lam}_b.pkl')
            self.result_dir = join(checkpoints_dir,f'{cl}{name}_{lam}_history.npy')


    def _eval_quantile(self, s, method_name='n_cdf'):
    #  from s~N(R^d) to u~(0,1)^d
    #  u_i = phi(s_i), where phi is the cdf of N(0,1)
        bs = s.shape[0]
        s_dim = s.shape[1]
        s_numpy = s.cpu().numpy()
        q_loss = []
        if method_name=='n_cdf':
            for i in range(bs):
                # for every sample
                # cdf 
                u = norm.cdf(s_numpy[i,:])
                u = np.ones((1,s_dim))*0.5-u
                u_q = np.linalg.norm(u,2)
                
                q_loss.append(u_q)
        else:
            ValueError("Unknown Mapping")

        return q_loss
    
    def _eval(self, x, average = True, quantile_flag = False):

        if self.name == 'LSA':
            # ok
            x_r = self.model(x)
            tot_loss = self.loss(x, x_r,average)

        elif self.name == 'LSA_EN':
            x_r, z, z_dist = self.model(x)
            tot_loss = self.loss(x, x_r, z, z_dist,average)
        
        elif self.name in ['LSA_SOS', 'LSA_MAF']:
            x_r, z, s, log_jacob_T_inverse = self.model(x)
            tot_loss = self.loss(x, x_r, s, log_jacob_T_inverse,average)
            # compute quantiles
            if quantile_flag:
                q_loss = self._eval_quantile(s)
                return tot_loss, q_loss

        elif self.name in ['SOS', 'MAF','LSA_ET_MAF','LSA_ET_SOS']:
            s, log_jacob_T_inverse = self.model(x)
            tot_loss = self.loss(s, log_jacob_T_inverse,average)
            if quantile_flag:
                q_loss= self._eval_quantile(s)
                return tot_loss, q_loss

        elif self.name in ['LSA_ET_EN']:
            z, z_dist = self.model(x)
            tot_loss = self.loss(z, z_dist) 

        return tot_loss

    def load_pretrained_model(self, model_name,cl):
        print(f"load pretraind")
        if self.pretrained:
            if model_name == 'LSA':
                
                self.model.load_state_dict(torch.load(f'checkpoints/{self.dataset.name}/combined{self.combined}/PtrFalse/{cl}LSA.pkl'),strict = False)
            elif model_name in ['LSA_SOS']:

                self.model.load_state_dict(torch.load(f'checkpoints/{self.dataset.name}/combined{self.combined}/PtrFalse/{cl}LSA.pkl'),strict = False)
                self.model.load_state_dict(torch.load(f'checkpoints/{self.dataset.name}/combined{self.combined}/PtrTrue/FixTrue/{cl}{model_name}.pkl'),strict = False)
            else:
                ValueError("Setting For New Pretrained Model")





    def add_two_pretrained_model(self, cl):
        ####load model1
        dataset_name = self.dataset.name
        lam = self.lam


        model_dict = self.model.state_dict()
        pretrained_model1 = LSA_MNIST(input_shape=self.dataset.shape, code_length=64, combine_density = False).cuda()
        pretrained_model1.load_w(f'checkpoints/{self.dataset.name}/combined{self.combined}/PtrFalse/{cl}LSA.pkl')
        print("Load the Pretrained LSA")

        pretrained_dict1= pretrained_model1.state_dict()
        pretrained_dict1 = {k: v for k, v in pretrained_dict1.items() if k in model_dict} 

        model_dict.update(pretrained_dict1) 
        self.model.load_state_dict(model_dict)

        ######load model2
        model_dict = self.model.state_dict()
        pretrained_model2 = LSAET_MNIST(input_shape=self.dataset.shape, code_length=64, num_blocks=1, est_name = 'SOS',hidden_size= 2048).cuda()
        print("Load the Pretrained LSA_ET")
        
        pretrained_model2.load_w(f'checkpoints/{self.dataset.name}/combined{self.combined}/PtrTrue/FixTrue/{cl}LSA_ET_SOS_{lam}.pkl') ## select best or last?

        pretrained_dict2= pretrained_model2.state_dict()
        pretrained_dict2 = {k: v for k, v in pretrained_dict2.items() if k in model_dict}

        # update & load
        model_dict.update(pretrained_dict2) 
        self.model.load_state_dict(model_dict)


    def train_every_epoch(self, epoch, cl):
      
            epoch_loss = 0
            epoch_recloss = 0
            epoch_nllk = 0

            # if self.dataset.name == 'mnist' and self.name in['LSA_SOS','LSA_MAF']:
            #     if epoch < 100:
            #         newlr =0.0001
            #     elif epoch <5000:
            #         newlr = 0.00001
            #     else:
            #         newlr  = 0.000001
            #     for param_group in self.optimizer.param_groups:
            #                 param_group['lr'] = newlr

         


            self.dataset.train(cl)
            loader = DataLoader(self.dataset, batch_size = self.batch_size,shuffle=True)

            epoch_size = self.dataset.length
            pbar = tqdm(total=epoch_size)
            s_alpha = 0

            for batch_idx, (x , y) in enumerate(loader):
                
                x = x.to(self.device)
                # self.model.zero_grad()
                self.optimizer.zero_grad()
                
                # backward average loss along batch
                if (self.mulobj) :
                # Multi-objective Optimization
                    self._eval(x)

                    torch.autograd.backward(self.loss.total_loss,self.model.parameters(),retain_graph =True)
                    #g1+g2
                    g1_list= [pi.grad.data.clone() for pi in list(self.model.encoder.parameters())]    
                    
                    torch.autograd.backward(self.loss.autoregression_loss,list(self.model.encoder.parameters())+list(self.model.estimator.parameters()))
                    #g2
                    g2_list= [pi.grad.data.clone() for pi in list(self.model.encoder.parameters())]
                    
                    for p in self.model.estimator.parameters():
                        p.grad.data.div(2)
                    # compute alpha
                    top = 0
                    down = 0
                    
                    i =0
                    for p in self.model.encoder.parameters():
                        g2   =  (g2_list[i]-g1_list[i])
                        g1   =  (g1_list[i]-g2)
                        g1 = g1.view(-1,)
                        g2 = g2.view(-1,)
                        
                        # g1   =  (g1_list[i]-g2).mean()
                        # g2   =  g2.mean()
                        # top  =  top + torch.mul((g2-g1),g2).item() 
                        
                        top   =  top + torch.mul((g2-g1),g2).sum()
                        down  =  down+ torch.pow((g1-g2),2).sum()
                        i     =  i + 1

                    if down ==0:
                        alpha =0.5
                    else:
                        alpha = (top/down)
                        alpha = max(min(alpha,1),0)
                        
                    
                    # compute new gradient of Shared Encoder
                    i=0

                    s_alpha =s_alpha + alpha*x.shape[0]
                    for p in self.model.encoder.parameters():
                        newlrg2 = g2_list[i]-g1_list[i]
                        newlrg1 = g1_list[i]-newlrg2
                        p.grad.zero_()
                        p.grad.data = newlrg1.mul(alpha)+newlrg2.mul(1-alpha)
                        # p.grad.data += newlrg1+newlrg2
                        i = i+1
                else:
                    self._eval(x).backward()

                
                self.optimizer.step()


                
                epoch_loss = + self.loss.total_loss.item()*x.shape[0]

                if self.name in ['LSA_EN','LSA_SOS','LSA_MAF']:
                    epoch_recloss =+ self.loss.reconstruction_loss.item()*x.shape[0]
                    epoch_nllk = + self.loss.autoregression_loss.item()*x.shape[0]

                pbar.update(x.size(0))
                pbar.set_description('Train, Loss: {:.6f}'.format(epoch_loss / (pbar.n)))


            pbar.close()

                # print epoch result
            if self.name in ['LSA_EN','LSA_SOS','LSA_MAF']:
                
                print('Train Epoch-{}: {}\tLoss: {:.6f}\tRec: {:.6f}\tNllk: {:.6f}'.format(
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

        self.dataset.val(cl)
        loader = DataLoader(self.dataset, self.batch_size)

        epoch_size = self.dataset.length
        batch_size = len(loader)

        # pbar = tqdm(total=epoch_size)
        # pbar.set_description('Eval')

        for batch_idx, (x,y) in enumerate(loader):
        
            x = x.to('cuda')
            # pbar.update(x.size(0))

            with torch.no_grad():
                loss =self. _eval(x,False)

                if self.name in ['LSA_EN','LSA_MAF','LSA_SOS']:
                    val_nllk += self.loss.autoregression_loss.sum().item()
                    val_rec += self.loss.reconstruction_loss.sum().item()
                    # keep lambda = 1
                    val_loss = val_nllk + val_rec
                else:
                     val_loss += self.loss.total_loss.sum().item() 
                                    
        if self.name in ['LSA_EN','LSA_MAF','LSA_SOS']:
            print('Val_loss:{:.6f}\t Rec: {:.6f}\t Nllk: {:.6f}'.format(val_loss/epoch_size, val_rec/epoch_size, val_nllk/epoch_size))
        else:
            print('Val_loss:{:.6f}\t'.format(val_loss/epoch_size))
            
            # pbar.set_description('Val_loss: {:.6f}'.format(val_loss))
        
        # pbar.close()
                

        return val_loss/epoch_size, val_rec/epoch_size,val_nllk/epoch_size






    def train_one_class_classification(self, cl):
        # type: () -> None
        """
        Actually performs trains.
        """     

        best_validation_epoch = 0
        best_validation_loss = float('+inf')
        best_model = None 
        old_validation_loss = float('+inf')

        history ={}
        history['val_loss'] =[]
        history['val_rec'] =[]
        history['val_nllk'] =[]

        history['trn_loss'] =[]
        history['trn_rec'] =[]
        history['trn_nllk'] =[]

        print(f"n_parameters:{self.model.n_parameters}")
        
        if self.pretrained:
            self.load_pretrained_model(self.pretrained_model,cl)

        for epoch in range(self.train_epoch):

            if self.mulobj:
                model_dir_epoch = join(self.checkpoints_dir,f'{cl}{self.name}_mul_{epoch}.pkl')

            else:
                model_dir_epoch = join(self.checkpoints_dir,f'{cl}{self.name}_{self.lam}_{epoch}.pkl')

            
            train_loss, train_rec, train_nllk= self.train_every_epoch(epoch,cl)
            
            # validate
            validation_loss,validation_rec,validation_nllk = self.validate(epoch, cl)

            old_validation_loss = validation_loss
            
               
            if (validation_loss < best_validation_loss): 
                best_validation_loss = validation_loss
                best_validation_epoch = epoch
                best_model = self.model 
                if (epoch>self.before_log_epochs):
                    torch.save(best_model.state_dict(), self.best_model_dir)

                print(f'Best_epoch at :{best_validation_epoch} with valid_loss:{best_validation_loss}, with lr:{self.lr}' )

            if (epoch % 200 == 0 ) :
                    torch.save(self.model.state_dict(), model_dir_epoch)
                    np.save(self.result_dir,history)
            
            # if (epoch -best_validation_epoch)>100 and (epoch>self.before_log_epochs):
            #     print (f"Break at Epoch:{epoch}")
            #     break
            
            # record loss history
            history['val_loss'].append(validation_loss)
            history['val_rec'].append(validation_rec)
            history['val_nllk'].append(validation_nllk)
            
            history['trn_loss'].append(train_loss)
            history['trn_rec'].append(train_rec)
            history['trn_nllk'].append(train_nllk)

            

        print("Training finish! Normal_class:>>>>>", cl)
        

        torch.save(self.model.state_dict(), self.model_dir)
        
        np.save(self.result_dir,history)

    def test_one_class_classification(self):
        # type: () -> None
        """
        Actually performs tests.
        """
        # Prepare a table to show results
        oc_table = self.empty_table

        # Set up container for metrics from all classes
        all_metrics = []
        bs =self.batch_size
        quantile_flag = self.quantile_flag

        # Start iteration over classes
        for cl_idx, cl in enumerate(self.dataset.test_classes):
            self.cl = cl
            self.get_path()
            print(f"Testinng on {cl}")

            if self.trainflag:
            # train model
                self.model.load_w(join(self.checkpoints_dir, f'{self.model.name}_start.pkl'))
                self.train_one_class_classification(cl)
            
            # Load the checkpoint    
            if self.add:
                self.add_two_pretrained_model(cl)
            else:
                self.model.load_w(self.model_dir)
                print(f"Load Model from {self.model_dir}")
            
            self.model.eval()

            if self.score_normed:
                # normalizing coefficient of the Novelty Score (Eq.9 in LSA)
                min_llk, max_llk, min_rec, max_rec = self.compute_normalizing_coefficients(cl)

            # Test sets
            self.dataset.test(cl,self.novel_ratio)
            data_num = len(self.dataset)
            loader = DataLoader(self.dataset, batch_size = bs)

            sample_llk = np.zeros(shape=(len(self.dataset),))
            sample_rec = np.zeros(shape=(len(self.dataset),))
            sample_qt = np.zeros(shape=(len(self.dataset),))
            sample_y = np.zeros(shape=(len(self.dataset),))

            for i, (x, y) in tqdm(enumerate(loader), desc=f'Computing scores for {self.dataset}'):
                
                x = x.to(self.device)

                with torch.no_grad():
                    if quantile_flag:
                       tot_loss, q_loss = self._eval(x, average = False, quantile_flag =quantile_flag)
                    else:
                        tot_loss = self._eval(x,average = False,quantile_flag = quantile_flag)
                
                sample_y[i*bs:i*bs+bs] = y
                # score larger-->normal data
                if self.name in ['LSA','LSA_MAF','LSA_SOS','LSA_EN','LSA_QT']:
                    sample_rec[i*bs:i*bs+bs] = - self.loss.reconstruction_loss.cpu().numpy()
                    
                if self.name in ['LSA_MAF','LSA_SOS','LSA_EN','LSA_QT',
                'EN','SOS','MAF',
                'LSA_ET_QT','LSA_ET_EN','LSA_ET_MAF','LSA_ET_SOS']:    
                    sample_llk[i*bs:i*bs+bs] = - self.loss.autoregression_loss.cpu().numpy()
                
                if quantile_flag:
                    sample_qt[i*bs:i*bs+bs] = q_loss

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
            
            if quantile_flag:
                sample_ns2 = novelty_score(sample_qt, sample_rec)
                sample_ns2 = modify_inf(sample_ns2)
                
            sample_ns = novelty_score(sample_llk, sample_rec)

            sample_ns = modify_inf(sample_ns)

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

            if self.name in ['LSA_EN','LSA_SOS','LSA_MAF','LSA_QT']:
                this_class_metrics.append(
                roc_auc_score(sample_y, sample_llk))
                
                this_class_metrics.append(
                roc_auc_score(sample_y, sample_rec))

            if self.name in ['LSA_EN','LSA_SOS','LSA_MAF','LSA_QT',
            'LSA_ET_EN','LSA_ET_SOS','LSA_ET_MAF','LSA_ET_QT']:

                this_class_metrics.append(llk1)
                this_class_metrics.append(llk2)
            
            # add metrics related to quantile 
            if quantile_flag:
                this_class_metrics.append(roc_auc_score(sample_y, sample_ns2))
                this_class_metrics.append(roc_auc_score(sample_y,sample_qt))


            # write on table
            oc_table.add_row([cl_idx] + this_class_metrics)

            all_metrics.append(this_class_metrics)

            print(f"Class_AUC-{cl}:{roc_auc_score(sample_y, sample_ns)}")



        # Compute average AUROC and print table
        all_metrics = np.array(all_metrics)
        avg_metrics = np.mean(all_metrics, axis=0)
        oc_table.add_row(['avg'] + list(avg_metrics))
        print(oc_table)

        # Save table
        with open(self.output_file, mode='w') as f:
            f.write(str(oc_table))

    
        
    def compute_normalizing_coefficients(self,cl):
        # type: (int) -> Tuple[float, float, float, float]
        """
        Computes normalizing coeffients for the computation of the Novelty score (Eq. 9-10).
        :param cl: the class to be considered normal.
        :return: a tuple of normalizing coefficients in the form (llk_min, llk_max, rec_min, rec_max).
        """
        bs = self.batch_size
        self.dataset.val(cl)
        loader = DataLoader(self.dataset, batch_size= bs)

        sample_llk = np.zeros(shape=(len(self.dataset),))
        sample_rec = np.zeros(shape=(len(self.dataset),))
        
        for i, (x, y) in enumerate(loader):
            
            x = x.to(self.device)
            with torch.no_grad():
                loss = self._eval(x,False)

            # score larger-->normal data
            if self.name in ['LSA','LSA_MAF','LSA_SOS','LSA_EN','LSA_QT']:
                sample_rec[i*bs:i*bs+bs] = - self.loss.reconstruction_loss.cpu().numpy()
            
            if self.name in ['LSA_MAF','LSA_SOS','LSA_EN','LSA_QT',
            'LSA_ET_EN','LSA_ET_MAF','LSA_ET_SOS','LSA_ET_QT',
            'EN','SOS','MAF']:    
                sample_llk[i*bs:i*bs+bs] = - self.loss.autoregression_loss.cpu().numpy()

            sample_llk = modify_inf(sample_llk)
            
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

            if self.quantile_flag:
                table.field_names = ['Class', 'AUROC-NS', 'AUROC-LLK', 'AUROC-REC','llk1','llk2'
                ,'AUROC-NSQ','AUROC-QT']
            else:
            
                table.field_names = ['Class', 'AUROC-NS', 'AUROC-LLK', 'AUROC-REC','llk1','llk2'
                ]
        elif self.name in ['MAF','SOS','EN','LSA',]:
            table.field_names = ['Class', 'AUROC-NS'
                ]

        elif self.name in ['LSA_ET_QT','LSA_ET_EN','LSA_ET_MAF','LSA_ET_SOS']:
            table.field_names = ['Class', 'AUROC-NS','llk1','llk2']
                
            
        elif self.name  in ['LSA_QT']:
            table.field_names = ['Class', 'AUROC-NS', 'AUROC-QT', 'AUROC-REC','q1','q2'
                ]

        # add new metric title related to quantile 
        

        # format
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
