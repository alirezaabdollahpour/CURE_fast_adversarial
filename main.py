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
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from models import *
from utils import *
from cure import *
from attacks.fmn import *

import warnings
warnings.filterwarnings('ignore')



logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--h', default=3.0, type=float, help='hyperparameter for CURE regulizer')
    parser.add_argument('--lambda_', default=4, type=float, help='weight for CURE regulizer')
    parser.add_argument('--gamma', default=0.5, type=float, help='weight for HAT loss')
    parser.add_argument('--betta', default=5.0, type=float, help='weight for TRADE loss')
    parser.add_argument('--kapa', default=1.0, type=float, help='weight for clean loss')
    parser.add_argument('--data-dir', default='cifar-data', type=str)
    parser.add_argument('--epochs', default=30, type=int)
    # parser.add_argument('--lr-schedule', default='multistep', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0.0, type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--lr-schedule', default='linear', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine','cyclic', 'fix', 'multistep', 'StepLR'])
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=10, type=float, help='Step size')
    parser.add_argument('--opt', default='SGD', type=str, choices=['SGD','Adam'], help='optimizer')
    parser.add_argument('--delta', default='linf', type=str, choices=['linf', 'random', 'classic', 'FGSM', 'None'] ,help='passing to CURE for FGSM direction rather thatn z')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous'],
        help='Perturbation initialization method')
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--early-stop', default=True, action='store_true', help='Early stop if overfitting occurs')
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    return parser.parse_args()


def main():
    args = get_args()
    gc.collect()
    state = {k: v for k, v in args._get_kwargs()}
    print(state)

    results_csv = f'train_FGSM_h_{args.h}_betta_{args.betta}_lambda_{args.lambda_}_epochs_{args.epochs}_lr_{args.lr_max}_lr_schedule_{args.lr_schedule}_{args.delta}_loss_clean.csv'
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader, train_dataset, test_dataset = get_loaders(args.data_dir, args.batch_size)

    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    pgd_alpha = (2 / 255.) / std

    # model = PreActResNet18().cuda()
    model = WideResNet().cuda()
    model.train()

    if args.opt == 'SGD':
        opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'Adam':
        opt = torch.optim.Adam(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)

    # Necessary for FP16
    scaler = torch.cuda.amp.GradScaler()
    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=True)
    criterion = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction='sum')
    cure = CURE(model, opt=opt, lambda_=args.lambda_)

    if args.delta_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()

    lr_steps = args.epochs * len(train_loader)
    ################# Learning rate schedule ####################################
    if args.lr_schedule == 'superconverge':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if t / args.epochs < 0.5:
                return args.lr_max
            elif t / args.epochs < 0.75:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.
    elif args.lr_schedule == 'linear':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100])[0]
    elif args.lr_schedule == 'onedrop':
        def lr_schedule(t):
            if t < args.lr_drop_epoch:
                return args.lr_max
            else:
                return args.lr_one_drop
    elif args.lr_schedule == 'multipledecay':
        def lr_schedule(t):
            return args.lr_max - (t//(args.epochs//10))*(args.lr_max/10)
    elif args.lr_schedule == 'cosine': 
        def lr_schedule(t): 
            return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))   
    elif args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max, step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[args.lr_max, args.lr_min], gamma=0.001)
    
    elif args.lr_schedule == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=10**6, gamma=1)

    regularizer_epochs = range(0, args.epochs+1,1)
    # Training
    prev_robust_acc = 0.
    start_train_time = time.time()
    accuracy_df = pd.DataFrame(columns=['epoch','loss_train','ACC_train','ACC_Clean_train','ACC_test'])
    for epoch in range(args.epochs):
        print("Start training")
        start_epoch_time = time.time()
        train_loss = 0
        test_loss  = 0
        train_acc = 0
        train_acc_clean = 0 
        test_acc = 0
        train_n = 0
        curvature = 0.0
        for i, (X, y) in tqdm(enumerate(train_loader)):
            X, y = X.cuda(), y.cuda()
            
            if args.lr_schedule != 'cyclic' and args.lr_schedule != 'fix' and args.lr_schedule != 'multistep' and args.lr_schedule != 'StepLR':
                lr = lr_schedule(epoch + (i + 1) / len(train_loader))
                opt.param_groups[0].update(lr=lr)
            
            if i == 0:
                first_batch = (X, y)

            if args.delta_init != 'previous':
                delta = torch.zeros_like(X).cuda()
            if args.delta_init == 'random':
                for j in range(len(epsilon)):
                    delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True
            with torch.cuda.amp.autocast():
                output = model(X + delta[:X.size(0)])
                loss = F.cross_entropy(output, y)


            scaler.scale(loss).backward()

            grad = delta.grad.detach()
            delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon) # FGSM
            delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            delta = delta.detach()
            
            # Forward pass with amp
            with torch.cuda.amp.autocast():
                ############ HAT #################
                X_adv_hat = X + delta[:X.size(0)]
                X_hr = X + 2 * (X_adv_hat - X)
                y_hr = model(X_hr).argmax(dim=1)
                out_help = model(X_hr)
                loss_help = F.cross_entropy(out_help, y_hr, reduction='mean')
                ##################################
                
                output = model(X + delta[:X.size(0)]) # gradient for theta to train with SGD or Adam
                loss = criterion(output, y)
                loss_robust = criterion_kl(F.log_softmax(output, dim=1),F.softmax(model(X), dim=1))
                logit_clean = model(X)
                loss_clean = criterion(logit_clean, y)
                     
                
            ########### CURE + TRADE ########################################
            if epoch in regularizer_epochs:
                ########### FMN_Linf for proper direction ##################
                 
                # Total loss : loss + TRADE_loss + CURE_regulizer

                if args.delta == 'linf':
                    best_adv, r_linf = fmn(model=model, inputs=X , labels=y, norm = 2.0, steps=5)
                    regularizer = cure.regularizer(X, y, delta='linf', h=args.h, X_adv=best_adv)
                    curvature += regularizer.item()
                    # Total loss : loss + TRADE_loss + CURE_regulizer
                    loss = loss + args.kapa*loss_clean + regularizer + (1/args.batch_size)*args.betta*loss_robust+args.gamma*loss_help
                elif args.delta == 'random':
                    regularizer = cure.regularizer(X, y, delta='random', h=args.h)
                    curvature += regularizer.item()
                    
                    loss = loss + args.kapa*loss_clean + regularizer + (1/args.batch_size)*args.betta*loss_robust+args.gamma*loss_help

                elif args.delta == 'classic':
                    regularizer = cure.regularizer(X, y, delta='classic', h=args.h)
                    curvature += regularizer.item()
                    
                    loss = loss + loss_clean + regularizer + (1/args.batch_size)*args.betta*loss_robust+args.gamma*loss_help

                elif args.delta == 'FGSM':
                    regularizer = cure.regularizer(X, y, delta='FGSM', h=args.h, X_adv=X +delta[:X.size(0)])
                    curvature += regularizer.item()
                    
                    loss = loss + loss_clean + regularizer + (1/args.batch_size)*args.betta*loss_robust+args.gamma*loss_help
                    
                elif args.delta == 'None':
                    loss = loss + loss_clean + (1/args.batch_size)*args.betta*loss_robust+args.gamma*loss_help
            
            else:
                loss = loss + loss_clean + (1/args.batch_size)*args.betta*loss_robust+args.gamma*loss_help
                
                

            opt.zero_grad()           
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_acc_clean += (logit_clean.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            
            if args.lr_schedule == 'cyclic':
                scheduler.step()  
            elif args.lr_schedule == 'multistep':
                scheduler.step()

            elif args.lr_schedule == 'StepLR':
                scheduler.step()
        
                
            
        if args.early_stop:
            # Check current PGD robustness of model using random minibatch
            X, y = first_batch
            pgd_delta = attack_pgd(model, X, y, epsilon, pgd_alpha, 20, 1, opt)
            with torch.no_grad():
                output = model(clamp(X + pgd_delta[:X.size(0)], lower_limit, upper_limit))
            robust_acc = (output.max(1)[1] == y).sum().item() / y.size(0)
            print("="*60)
            print(f'Robust Accurcy for epoch:{epoch} is :{robust_acc*100}%')
            print("="*60)
            if robust_acc - prev_robust_acc < -0.2:
                break
            prev_robust_acc = robust_acc
            best_state_dict = copy.deepcopy(model.state_dict())
            test_acc = robust_acc
        epoch_time = time.time()
        ############## Print Information ##########################################
        print(f'curvature for epoch:{epoch} is :{curvature}')
        print("="*60)
        print(f'Train Accuracy for epoch:{epoch} is :{(train_acc/train_n)*100}%')
        print("="*60)
        print(f'Train Accuracy on clean samples for epoch:{epoch} is :{(train_acc_clean/train_n)*100}%')
        print("="*60)
        epoch_time = time.time()
        print('Total epoch time: %.4f minutes', (epoch_time - start_epoch_time)/60)
        ###########################################################################
        accuracy_df = accuracy_df._append({'epoch': epoch, 'loss_train': train_loss/train_n ,'ACC_train':(train_acc/train_n)*100,'ACC_Clean_train':(train_acc_clean/train_n)*100,'ACC_test':test_acc*100}, ignore_index=True)

        # lr = scheduler.get_lr()[0]

    # Save the DataFrame to a CSV file
    accuracy_df.to_csv(results_csv, index=False)
    train_time = time.time()
    if not args.early_stop:
        best_state_dict = model.state_dict()
    torch.save(best_state_dict, os.path.join(args.out_dir, 'model.pth'))

    # Evaluation
    # model_test = PreActResNet18().cuda()
    model_test = WideResNet().cuda()
    model_test.load_state_dict(best_state_dict)
    model_test.float()
    model_test.eval()

    # Select 1024 random indices from the test set
    indices = torch.randperm(len(test_dataset))[:1024]
    # Create the subset
    test_subset = Subset(test_dataset, indices)
    testloader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
    pgd_loss, pgd_acc = evaluate_pgd(testloader, model_test, 20, 1) # zero-restarts
    print('='*60)
    print(f'pgd_loss is :{pgd_loss} and pgd_acc(robust acc) is :{pgd_acc*100}')
    print('='*60)
    ACC_AA_PGD = evaluate_robust_accuracy_AA_APGD(model_test, testloader, 'cuda', epsilon=8/255)
    test_loss, test_acc = evaluate_standard(testloader, model_test)
    print(f'Robust accuray for AA-PGD is :{ACC_AA_PGD}')
    print('='*60)
    print(f'test_loss is :{test_loss} and clean_acc is:{test_acc*100}')





if __name__ == "__main__":
    main()