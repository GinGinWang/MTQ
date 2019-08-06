import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import math



def create_checkpoints_dir(dataset, num_blocks, hidden_size, code_length, estimator, noise = 0, noise2= 0, noise3 = 0, noise4 = 0):
    
    dirName = f'checkpoints/{dataset}/'
    
    if estimator == 'SOS':
        dirName = f'{dirName}b{num_blocks}h{hidden_size}c{code_length}/'
    
    if noise >0:
        dirName = f'{dirName}noise_{noise}/'
    elif noise2 >0:
        dirName = f'{dirName}noise2_{noise2}/'
    elif noise3 >0:
        dirName = f'{dirName}noise3_{noise3}/'
    elif noise4 >0:
        dirName = f'{dirName}noise4_{noise4}/'

    #create dir
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print(f'Make Dir:{dirName}')

    return dirName

def create_file_path(mulobj, model_name, dataset, score_normed, novel_ratio, num_blocks, hidden_size, code_length, lam, noise, noise2, noise3, noise4, checkpoint):

    if mulobj:
        result_path = f"results/Mul_{model_name}_{dataset}_nml{score_normed}_nlr{novel_ratio}_b{num_blocks}_h{hidden_size}_c{code_length}_lam{lam}"
    else:
       result_path = f"results/{model_name}_{dataset}_nml{score_normed}_nlr{novel_ratio}_b{num_blocks}_h{hidden_size}_c{code_length}_lam{lam}"

    
    if (noise > 0):
        result_path = f"{result_path}_ns{noise}"

    elif (noise2 > 0):
        result_path = f"{result_path}_ns2{noise2}"
    
    elif(noise3>0):
        result_path = f"{result_path}_ns3{noise3}"

    elif(noise4>0):
        result_path = f"{result_path}_ns4{noise4}"
    
    if (checkpoint!=None):
        result_path = f"{result_path}_at{checkpoint}"

    result_path = f"{result_path}.txt"

    return result_path


def set_random_seed(seed):
    # type: (int) -> None
    """
    Sets random seeds.
    :param seed: the seed to be set for all libraries.
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic= True
    
    random.seed(seed)
    np.random.seed(seed)
    # np.random.shuffle.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



# def weights_init(model):
    

    

    # for m in model.modules():

    #     for p in m.parameters():
    #         p.data.fill_(0)

    #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    #         init.orthogonal_(m.weight)
    #         if m.bias:
    #             init.orthogonal_(m.bias)

    #         # n = m.in_channels
    #         # for k in m.kernel_size:
    #         #     n *= k
    #         #     stdv = 1. / math.sqrt(n)
    #         # m.weight.data.uniform_(-stdv, stdv)
    #         # if m.bias is not None:
    #         #     m.bias.data.uniform_(-stdv, stdv)

    #     # for fashion mnist
    #     elif isinstance(m, nn.BatchNorm2d):
    #         if m.track_running_stats:
    #             m.running_mean.zero_()
    #             m.running_var.fill_(1)
    #         if m.affine:
    #             init.ones_(m.weight)
    #             m.bias.data.zero_()

    #     # for fashion mnist
    #     elif isinstance(m, nn.BatchNorm1d):
    #         if m.track_running_stats:
    #             m.running_mean.zero_()
    #             m.running_var.fill_(1)
    #         if m.affine:
    #             init.ones_(m.weight)
    #             m.bias.data.zero_()

    #     elif isinstance(m, nn.Linear):
    #         # stdv = 1. / math.sqrt(m.weight.size(1))
    #         # m.weight.data.uniform_(-stdv, stdv)
    #         # if m.bias is not None:
    #         #     m.bias.data.uniform_(-stdv, stdv)

    #         # mnist
    #         nn.init.ones_(m.weight)
    #         if m.bias is not None:
    #             init.ones_(m.bias)


            
            

            



def weights_init(m):
    
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)