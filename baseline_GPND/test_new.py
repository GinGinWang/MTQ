# Copyright 2018 Stanislav Pidhorskyi
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import print_function
import torch.utils.data
from torch import optim
from torchvision.utils import save_image
from net import *
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import json
import pickle
import time
import random
from torch.autograd.gradcheck import zero_gradients
import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt
import scipy.stats
import os
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from datasets.cifar10 import CIFAR10
from datasets.mnist import MNIST
from datasets.fmnist import FMNIST


title_size = 16
axis_title_size = 14
ticks_size = 18

power = 2.0

device = torch.device("cuda")
use_cuda = torch.cuda.is_available()

FloatTensor = torch.FloatTensor
IntTensor = torch.IntTensor
LongTensor = torch.LongTensor
torch.set_default_tensor_type('torch.FloatTensor')

if use_cuda:
    device = torch.cuda.current_device()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    FloatTensor = torch.cuda.FloatTensor
    IntTensor = torch.cuda.IntTensor
    LongTensor = torch.cuda.LongTensor
    print("Running on ", torch.cuda.get_device_name(device))


def setup(x):
    if use_cuda:
        return x.cuda()
    else:
        return x.cpu()


def numpy2torch(x):
    return setup(torch.from_numpy(x))

def compute_jacobian(inputs, output):
    """
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :param output: Batch X Classes
    :return: jacobian: Batch X Classes X Size
    """
    assert inputs.requires_grad

    num_classes = output.size()[1]

    jacobian = torch.zeros(num_classes, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_graph=True)
        jacobian[i] = inputs.grad.data

    return torch.transpose(jacobian, dim0=0, dim1=1)

def main( cl, dataset_name, noise, run):

    if dataset_name == 'cifar10':
        dataset = CIFAR10(path='data/')
        channels = 3
        z_size = 256
    elif dataset_name == 'fmnist':
        dataset = FMNIST(path= 'data/')
        channels = 1 
        z_size = 64
 
    elif dataset_name == 'mnist':
        dataset = MNIST(path = 'data/')
        channels = 1
        z_size = 16
 
    
    batch_size = 128

    

    G = Generator(z_size = z_size, channels = channels).to(device)
    E = Encoder(code_length = z_size, channels= channels).to(device)
    setup(E)
    setup(G)
    G.eval()
    E.eval()

    if noise == 0:
        G.load_state_dict(torch.load(f"{dataset_name}_c{cl}_z{z_size}_G.pkl"))
        E.load_state_dict(torch.load(f"{dataset_name}_c{cl}_z{z_size}_E.pkl"))
    else:
        G.load_state_dict(torch.load(f"{dataset_name}_n{noise}_z{z_size}_G_run{run}.pkl"))
        E.load_state_dict(torch.load(f"{dataset_name}_n{noise}_z{z_size}_E_run{run}.pkl"))
        
    if True:
        zlist = []
        rlist = []


        dataset.train(normal_class = cl, noise_ratio= noise)

        loader = DataLoader(dataset, batch_size = batch_size)
        
        
        for batch_idx, (x , y) in enumerate(loader):
            
            print(f"cl{cl}:batch{batch_idx}")
            bs = x.shape[0]
            x = setup(x)

            z = E(x.view(-1, channels, 32, 32))
            recon_batch = G(z)
            z = z.squeeze()
            recon_batch = recon_batch.squeeze().cpu().detach().numpy()
            x = x.squeeze().cpu().detach().numpy()
            x = x.squeeze()
            z = z.cpu().detach().numpy()

            for i in range(bs):
                distance = np.sum(np.power(recon_batch[i].flatten() - x[i].flatten(), power))
                rlist.append(distance.item())

            zlist.append(z) # hidden vector

        data = {}
        data['rlist'] = rlist # distance
        print(len(rlist)) 
        data['zlist'] = zlist # hidden vector
        print(len(zlist))
        

        with open('data{dataset.name}.pkl', 'wb') as pkl:
            pickle.dump(data, pkl)
            print("list saved!")

    with open('data{dataset.name}.pkl', 'rb') as pkl:
        data = pickle.load(pkl)
        print("list loded ")

    rlist = data['rlist']
    zlist = data['zlist']

    counts, bin_edges = np.histogram(rlist, bins=30, normed=True)


    def r_pdf(x, bins, count):
        if x < bins[0]:
            return max(count[0], 1e-308)
        if x >= bins[-1]:
            return max(count[-1], 1e-308)
        id = np.digitize(x, bins) - 1
        return max(count[id], 1e-308)

    zlist = np.concatenate(zlist)
    

    gennorm_param = np.zeros([3, z_size])
    
    for i in range(z_size):
        betta, loc, scale = scipy.stats.gennorm.fit(zlist[:, i])
        gennorm_param[0, i] = betta
        gennorm_param[1, i] = loc
        gennorm_param[2, i] = scale

    
    def test():
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        count = 0

        result = []

        dataset.test(cl)
        print(f"Test-{cl}")

        loader2 = DataLoader(dataset, batch_size = batch_size)
        print("Datastream")
        for batch_idx, (x , label) in enumerate(loader2):

            bs = x.shape[0]
            x = setup(x)
            x = Variable(x, requires_grad= True)
        
            z = E(x.view(-1, channels, 32, 32))
            recon_batch = G(z)
            z = z.squeeze()

            J = compute_jacobian(x, z)

            J = J.cpu().numpy()

            z = z.cpu().detach().numpy()

            recon_batch = recon_batch.squeeze().cpu().detach().numpy()
            x = x.squeeze().cpu().detach().numpy()

            for i in range(bs):
                print(f"TEST{cl}:{batch_idx}-{i}")
                u, s, vh = np.linalg.svd(J[i, :, :], full_matrices=False)
                logD = np.sum(np.log(np.abs(s)))

                p = scipy.stats.gennorm.pdf(z[i], gennorm_param[0, :], gennorm_param[1, :], gennorm_param[2, :])
                logPz = np.sum(np.log(p))

                # Sometimes, due to rounding some element in p may be zero resulting in Inf in logPz
                # In this case, just assign some large negative value to make sure that the sample 
                # is classified as unknown. 
                if not np.isfinite(logPz):
                    logPz = -1000

                distance = np.sum(np.power(x[i].flatten() - recon_batch[i].flatten(), power))

                logPe = np.log(r_pdf(distance, bin_edges, counts))
                logPe -= np.log(distance) * (32 * 32 - z_size)

                count += 1

                P = logD + logPz + logPe

                result.append(((label[i].item()==1), P))


        y_true = [x[0] for x in result]
        y_scores = [x[1] for x in result]

        try:
            auc = roc_auc_score(y_true, y_scores)

        except:
            auc = 0

        print("AUC ", auc)

        result_path = f"results_{dataset.name}_z{z_size}_n{noise}_run{run}.txt"
        with open(os.path.join(result_path), "a") as file:
            file.write(
                "Class: %d\n AUC: %f\n " %
                (cl, auc))
        return auc

    results = test()
    return results

if __name__ == '__main__':
    
    noNoise = False
    noise_list = [0.01,0.02,0.03,0.04,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
    result = []
    dataset_name = 'mnist'
    if noNoise:
        for i in range(0,10):
            print(f"Class{i}")
            result.append(main(i,dataset_name))
    else: 
        # only have two class
        for run in range(5):
            for noise in noise_list:
                print(f"test for noise:{noise}")
                main(8, dataset_name, noise, run)
                print (result) 