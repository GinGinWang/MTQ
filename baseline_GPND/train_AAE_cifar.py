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
import os
from torch.utils.data import DataLoader
from datasets.cifar10 import CIFAR10


use_cuda = torch.cuda.is_available()

FloatTensor = torch.FloatTensor
IntTensor = torch.IntTensor
LongTensor = torch.LongTensor
torch.set_default_tensor_type('torch.FloatTensor')

# If zd_merge true, will use zd discriminator that looks at entire batch.
zd_merge = False

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


def extract_batch(data, it, batch_size):
    x = numpy2torch(data[it * batch_size:(it + 1) * batch_size, :, :]) / 255.0
    #x.sub_(0.5).div_(0.5)
    return Variable(x)


def main(inliner_classes, total_classes):
    cl = inliner_classes[0]
    print("Training for {cl}")
    batch_size = 128
    zsize = 32
    


    # mnist_train = []
    # mnist_valid = []

    # for i in range(folds):
    #     if i != folding_id:
    #         with open('data_fold_%d.pkl' % i, 'rb') as pkl:
    #             fold = pickle.load(pkl)
    #         if len(mnist_valid) == 0:
    #             mnist_valid = fold
    #         else:
    #             mnist_train += fold
    # with open('traindata.pkl', 'rb') as pkl:
    #     mnist_train = pickle.load(pkl)
    # with open('validdata.pkl', 'rb') as pkl:
    #     mnist_valid = pickle.load(pkl)
    dataset = CIFAR10(path='data/')
    dataset.train(cl)

    outlier_classes = []
    for i in range(total_classes):
        if i not in inliner_classes:
            outlier_classes.append(i)

    #keep only train classes
    # mnist_train = [x for x in mnist_train if x[0] in inliner_classes]

    # random.shuffle(mnist_train)

    # def list_of_pairs_to_numpy(l):
    #     return np.asarray([x[1] for x in l], np.float32), np.asarray([x[0] for x in l], np.int)

    # print("Train set size:", len(mnist_train))

    # mnist_train_x, mnist_train_y = list_of_pairs_to_numpy(mnist_train)

    G = Generator(z_size = zsize, channels = 3)
    setup(G)
    G.weight_init(mean=0, std=0.02)

    D = Discriminator(channels = 3)
    setup(D)
    D.weight_init(mean=0, std=0.02)

    E = Encoder(code_length = zsize, channels =3)
    setup(E)
    E.weight_init(mean=0, std=0.02)

    if zd_merge:
        ZD = ZDiscriminator_mergebatch(zsize, batch_size).to(device)
    else:
        ZD = ZDiscriminator(zsize, batch_size).to(device)

    setup(ZD)
    ZD.weight_init(mean=0, std=0.02)

    lr = 0.002

    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    E_optimizer = optim.Adam(E.parameters(), lr=lr, betas=(0.5, 0.999))
    GE_optimizer = optim.Adam(list(E.parameters()) + list(G.parameters()), lr=lr, betas=(0.5, 0.999))
    ZD_optimizer = optim.Adam(ZD.parameters(), lr=lr, betas=(0.5, 0.999))

    train_epoch = 80

    BCE_loss = nn.BCELoss()
    sample = torch.randn(64, zsize).view(-1, zsize, 1, 1)

    for epoch in range(train_epoch):
        G.train()
        D.train()
        E.train()
        ZD.train()

        Gtrain_loss = 0
        Dtrain_loss = 0
        Etrain_loss = 0
        GEtrain_loss = 0
        ZDtrain_loss = 0

        epoch_start_time = time.time()

        # def shuffle(X):
        #     np.take(X, np.random.permutation(X.shape[0]), axis=0, out=X)

        # shuffle(mnist_train_x)

        if (epoch + 1) % 30 == 0:
            G_optimizer.param_groups[0]['lr'] /= 4
            D_optimizer.param_groups[0]['lr'] /= 4
            GE_optimizer.param_groups[0]['lr'] /= 4
            E_optimizer.param_groups[0]['lr'] /= 4
            ZD_optimizer.param_groups[0]['lr'] /= 4
            print("learning rate change!")
   
        # for it in range(len(mnist_train_x) // batch_size):
        #     x = extract_batch(mnist_train_x, it, batch_size).view(-1, 1, 32, 32)
            #############################################
        loader = DataLoader(dataset, batch_size = batch_size,shuffle=True)
        
        for batch_idx, (x , y) in enumerate(loader):
            bs = x.shape[0]
            x = setup(x)
            D.zero_grad()

            D_result = D(x).squeeze()
            y_real_ = torch.ones(D_result.shape[0])
            D_real_loss = BCE_loss(D_result, y_real_)

            z = torch.randn((batch_size, zsize)).view(-1, zsize, 1, 1)
            z = Variable(z)

            x_fake = G(z).detach()
            D_result = D(x_fake).squeeze()
            # print(x_fake.shape)
            # print(D_result.shape)
            # print(y_fake_.shape)
            
            y_fake_ = torch.zeros(D_result.shape[0])
            D_fake_loss = BCE_loss(D_result, y_fake_)
            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()

            D_optimizer.step()

            Dtrain_loss += D_train_loss.item()

            #############################################

            G.zero_grad()

            z = torch.randn((batch_size, zsize)).view(-1, zsize, 1, 1)
            z = Variable(z)

            x_fake = G(z)
            D_result = D(x_fake).squeeze()
            y_real_ = torch.ones(1 if zd_merge else D_result.shape[0])
            G_train_loss = BCE_loss(D_result, y_real_)

            G_train_loss.backward()
            G_optimizer.step()

            Gtrain_loss += G_train_loss.item()

            #############################################

            ZD.zero_grad()

            z = torch.randn((batch_size, zsize)).view(-1, zsize)
            z = Variable(z)

            ZD_result = ZD(z).squeeze()
            y_real_z = torch.ones(1 if zd_merge else ZD_result.shape[0])
            ZD_real_loss = BCE_loss(ZD_result, y_real_z)

            z = E(x).squeeze().detach()

            ZD_result = ZD(z).squeeze()
            y_fake_z = torch.zeros(1 if zd_merge else ZD_result.shape[0])
            ZD_fake_loss = BCE_loss(ZD_result, y_fake_z)

            ZD_train_loss = ZD_real_loss + ZD_fake_loss
            ZD_train_loss.backward()

            ZD_optimizer.step()

            ZDtrain_loss += ZD_train_loss.item()

            #############################################

            E.zero_grad()
            G.zero_grad()

            z = E(x)
            x_d = G(z)

            ZD_result = ZD(z.squeeze()).squeeze()

            y_real_z = torch.ones(1 if zd_merge else ZD_result.shape[0])
            E_loss = BCE_loss(ZD_result, y_real_z) * 2.0

            Recon_loss = F.binary_cross_entropy(x_d, x)

            (Recon_loss + E_loss).backward()

            GE_optimizer.step()

            GEtrain_loss += Recon_loss.item()
            Etrain_loss += E_loss.item()

            # if it == 0:
            #     directory = 'results'+str(inliner_classes[0])
            #     if not os.path.exists(directory):
            #         os.makedirs(directory)
            #     comparison = torch.cat([x[:64], x_d[:64]])
            #     save_image(comparison.cpu(),
            #                'results'+str(inliner_classes[0])+'/reconstruction_' + str(epoch) + '.png', nrow=64)

        Gtrain_loss /= (dataset.length)
        Dtrain_loss /= (dataset.length)
        ZDtrain_loss /= (dataset.length)
        GEtrain_loss /= (dataset.length)
        Etrain_loss /= (dataset.length)

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        print('[%d/%d] - ptime: %.2f, Gloss: %.3f, Dloss: %.3f, ZDloss: %.3f, GEloss: %.3f, Eloss: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, Gtrain_loss, Dtrain_loss, ZDtrain_loss, GEtrain_loss, Etrain_loss))

        # with torch.no_grad():
        #     resultsample = G(sample).cpu()
        #     directory = 'results'+str(inliner_classes[0])
        #     os.makedirs(directory, exist_ok = True)
        #     save_image(resultsample.view(64, 1, 32, 32), 'results'+str(inliner_classes[0])+'/sample_' + str(epoch) + '.png')


    print("Training finish!... save training results")
    torch.save(G.state_dict(), f"{cl}cifar_Gmodel.pkl")
    torch.save(E.state_dict(), f"{cl}cifar_Emodel.pkl")
    torch.save(D.state_dict(), f"{cl}cifar_Dmodel.pkl")
    torch.save(ZD.state_dict(),f"{cl}cifar_ZDmodel.pkl")

if __name__ == '__main__':
    for i in range(10):
        main([i], 10)