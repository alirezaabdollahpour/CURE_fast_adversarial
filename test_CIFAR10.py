import argparse
import copy
import logging
import os
import time
import gc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from models import *
from utils import *
from main import *
from cure import *
from preactresnet import *
# from resnet import *

import warnings
warnings.filterwarnings('ignore')


def attack_pgd_Alireza(model, X, y, epsilon, alpha, attack_iters, restarts, norm, early_stop=False, fgsm_init=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if attack_iters>1 or fgsm_init=='random': 
            if norm == "l_inf":
                delta.uniform_(-epsilon, epsilon)
            elif norm == "l_2":
                delta.normal_()
                d_flat = delta.view(delta.size(0),-1)
                n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
                r = torch.zeros_like(n).uniform_(0, 1)
                delta *= r/n*epsilon
            else:
                raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        with torch.no_grad():
            all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
            max_delta[all_loss >= max_loss] = torch.clone(delta.detach()[all_loss >= max_loss])
            max_loss = torch.max(max_loss, all_loss)
    return max_delta

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--data-dir', default='cifar-data', type=str)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=2, type=float, help='Step size')
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--eval', action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    gc.collect()
    state = {k: v for k, v in args._get_kwargs()}
    print(state)

    results_csv = 'AutoAttack_PGD_20_full_epsilon_RN18.csv'
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader, train_dataset, test_dataset = get_loaders(args.data_dir, args.batch_size)

    # epsilon = (args.epsilon / 255.) / std
    # alpha = (args.alpha / 255.) / std
    # pgd_alpha = (2 / 255.) / std
    
    # Evaluation
    model_test = PreActResNet18()
    
    model_test.load_state_dict(torch.load('train_fgsm_output/model_best_PRN18_lamda_900_epsilon_16_g3_direction_1_10_alpha_4.pth'))
    # model_test = WideResNet().cuda()
    # model_test = resnet(name='resnet18', num_classes=10).cuda()

    # model_test.float()
    # model_test.eval()
    model_test.cuda()
    model_test.eval()

    metrics = pd.DataFrame(columns=['epsilon','ACC_PGD','ACC_AA'])
    # Select 1024 random indices from the test set
    indices = torch.randperm(len(test_dataset))[:9000]
    # Create the subset
    test_subset = Subset(test_dataset, indices)
    testloader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
    # test_loss, test_acc = evaluate_standard(testloader, model_test)
    # print(f'test_loss is :{test_loss} and clean_acc is:{test_acc*100}')
    # Epsilons = range(2,20,2)
    Epsilons = [16.0]
    # pgd_alpha = (10 / 255.)
    criterion = nn.CrossEntropyLoss()
    for epsilon in Epsilons:
        
        epsilon = (epsilon/255.)
        pgd_alpha = (4./255.)
        print(f'pgd_alpha is :{pgd_alpha}')
        test_loss = 0
        test_acc = 0
        test_robust_loss = 0
        test_robust_acc = 0
        test_n = 0
        
        for i, (X, y) in tqdm(enumerate(testloader)):
            X = X.cuda()
            y = y.cuda()
            # delta = attack_pgd_Alireza(model_test, X, y, epsilon, pgd_alpha, 20, 1, 'l_inf', early_stop=False, fgsm_init=None)
            # delta = delta.detach()
            # robust_output = model_test(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
            # robust_loss = criterion(robust_output, y)
            # clean_output = model_test(normalize(X))
            # clean_loss = criterion(clean_output, y)

            # test_robust_loss += robust_loss.item() * y.size(0)
            # test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            # test_loss += clean_loss.item() * y.size(0)
            # test_acc += (clean_output.max(1)[1] == y).sum().item()
            # test_n += y.size(0)
            # print("="*80)
            # print(f'Test Accuracy on PGD samples for epsilon:{epsilon} is :{(test_robust_acc/test_n)*100}%')
            # print("="*80)
            # print(f'Test Accuracy on clean samples is :{(test_acc/test_n)*100}%')
            # print("="*80)   
            # print('='*50)
            # print(f'pgd_loss for epsilon={epsilon} is :{pgd_loss} and pgd_acc(robust acc) for epsilon={epsilon} is :{pgd_acc*100}')
            print('='*50)
            ACC_AA = evaluate_robust_accuracy_AA_Complete(model_test, testloader, 'cuda', epsilon=epsilon)     
            print(f'Robust accuray for AA at epsilon={epsilon} is :{ACC_AA}')
            print('='*50)
        # metrics = metrics._append({'epsilon': epsilon, 'ACC_PGD': pgd_acc,'ACC_AA':ACC_AA}, ignore_index=True)
        
    # metrics.to_csv(results_csv, index=False)

        




if __name__ == "__main__":
    main()