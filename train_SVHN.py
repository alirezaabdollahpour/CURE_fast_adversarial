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

# from autoattack import AutoAttack
# from models import *
# from resnet import *
from utils import str_to_bool, clamp
from torchvision import datasets, transforms
from cure import *
from preactresnet import *
# from linearize import *
# from attacks.fmn import *

import warnings
warnings.filterwarnings('ignore')


svhn_mean = (0.5, 0.5, 0.5)
svhn_std = (0.5, 0.5, 0.5)

mu_SVHN = torch.tensor(svhn_mean).view(3,1,1).cuda()
std_SVHN = torch.tensor(svhn_std).view(3,1,1).cuda()

upper_limit, lower_limit = 1, 0

def get_loaders_SVHN(dir_, batch_size):
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    num_workers = 2
    train_dataset = datasets.SVHN(
        dir_, split='train', transform=train_transform, download=True)
    test_dataset = datasets.SVHN(
        dir_, split='test', transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    
    return train_loader, test_loader, train_dataset, test_dataset


def normalize(X):
    return (X - mu_SVHN)/std_SVHN



def zero_grad(model, X, y, epsilon, alpha, q_val, q_iters, fgsm_init):
    delta = torch.zeros_like(X)
    full_delta = torch.zeros_like(X)
    if fgsm_init=='random':
        delta.uniform_(-epsilon, epsilon)
        delta = clamp(delta, lower_limit - X, upper_limit - X)
    for i in range(q_iters):
        delta.requires_grad = True
        full_delta.requires_grad = True
        with torch.cuda.amp.autocast():
            output = model(normalize(X + delta))
            F.cross_entropy(output, y).backward()
        grad = delta.grad.detach()
        ######### gradient without masking ################
        full_grad = copy.deepcopy(grad)
        ###################################################
        q_grad = torch.quantile(torch.abs(grad).view(grad.size(0), -1), q_val, dim=1)
        grad[torch.abs(grad) < q_grad.view(grad.size(0), 1, 1, 1)] = 0
        delta = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
        ##########################################################################################
        full_delta = torch.clamp(delta + alpha * torch.sign(full_grad), min=-epsilon, max=epsilon)
        ###########################################################################################
        delta = clamp(delta, lower_limit - X, upper_limit - X)
        ###########################################################################################
        full_delta = clamp(full_delta, lower_limit - X, upper_limit - X)
        ###########################################################################################
        delta = delta.detach()
        full_delta = full_delta.detach()
    return delta.detach(), full_delta.detach()
    # return delta.detach()

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
    parser.add_argument('--h', default=0.00, type=float, help='hyperparameter for CURE regulizer')
    parser.add_argument('--lambda_', default=10, type=float, help='weight for CURE regulizer')
    parser.add_argument('--gamma', default=0.00, type=float, help='weight for HAT loss')
    parser.add_argument('--betta', default=0.00, type=float, help='weight for TRADE loss')
    parser.add_argument('--kapa', default=0.00, type=float, help='weight for clean loss')
    parser.add_argument('--data-dir', default='SVHN-data', type=str)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--attack', default='zerograd', type=str, choices=['zerograd', 'fgsm', 'pgd'])
    parser.add_argument('--lr-min', default=0.0, type=float)
    parser.add_argument('--lr-max', default=0.05, type=float)
    parser.add_argument('--lr-schedule', default='cyclic', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine','cyclic', 'fix', 'multistep', 'StepLR'])
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)
    parser.add_argument('--zero-qval', default=0.70, type=float, help='quantile which gradients would become zero in zerograd')
    parser.add_argument('--zero-iters', default=1, type=int, help='number of zerograd iterations')
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float, help='pgd alpha WONT be multiplied by epsilon')
    parser.add_argument('--fgsm-alpha', default=2, type=float, help='fgsm alpha WILL be multiplied by epsilon')
    parser.add_argument('--opt', default='SGD', type=str, choices=['SGD','Adam'], help='optimizer')
    parser.add_argument('--hat', type=str_to_bool, default=False, help='Using HAT loss (0 or 1)')
    parser.add_argument('--interpol', type=str_to_bool, default=False, help='Using interpolation model (0 or 1)')
    parser.add_argument('--delta', default='FGSM', type=str, choices=['linf', 'random', 'classic', 'FGSM', 'None'] ,help='passing to CURE for FGSM direction rather thatn z')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous'],
        help='Perturbation initialization method')
    parser.add_argument('--out-dir', default='SVHN_Models', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--early-stop', default=True, action='store_true', help='Early stop if overfitting occurs')
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')

    parser.add_argument('--test-iters', default=20, type=int, help='number of pgd steps used in evaluation during training')
    parser.add_argument('--test-restarts', default=1, type=int, help='number of pgd restarts used in evaluation during training')
    
    return parser.parse_args()


def main():
    args = get_args()
    gc.collect()
    state = {k: v for k, v in args._get_kwargs()}
    print(state)

    results_csv = f'SVHN_lambda_{args.lambda_}_epochs_{args.epochs}_lr_{args.lr_max}_lr_schedule_{args.lr_schedule}_{args.delta}_loss_clean_g3_direction_{args.attack}_epsilon_{args.epsilon}_FGSM_alpha_{args.fgsm_alpha}_threshold_{args.zero_qval}.csv'
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader, train_dataset, test_dataset = get_loaders_SVHN(args.data_dir, args.batch_size)

    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)

    model = PreActResNet18(num_classes=10).cuda()
    # model = DMPreActResNet()
    # model = LinearizedModel(model, init_model=model)
    # model = model.cuda()
    # model = resnet(name='resnet18', num_classes=10).cuda()
    # model = resnet()
    # Teacher = resnet(name='resnet18', num_classes=10)
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
        scheduler = lambda t: np.interp([t], [0, args.epochs // 2, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[args.lr_max, args.lr_min], gamma=0.001)
    
    elif args.lr_schedule == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=10**6, gamma=1)
    
    ################################################################################################################

    regularizer_epochs = range(0, args.epochs+1,1)
    # interpolation_epochs = range(0, args.epochs+1,1)
    # Training
    prev_robust_acc = 0.
    best_test_robust_acc = 0.
    start_train_time = time.time()
    accuracy_df = pd.DataFrame(columns=['epoch','loss_train_FGSM','loss_train_clean','ACC_train','ACC_Clean_train','ACC_test_clean','Acc_test_PGD','Curvature','grad_norm','test_loss_clean'])
    for epoch in range(args.epochs):
        print("Start training")
        start_epoch_time = time.time()
        train_loss = 0.0
        train_loss_clean = 0.0
        test_loss  = 0.0
        train_acc = 0.0
        train_acc_clean = 0.0 
        test_acc = 0.0
        train_n = 0.0
        curvature = 0.0
        gradient_norm = 0.0
        for i, (X, y) in tqdm(enumerate(train_loader)):
            X, y = X.cuda(), y.cuda()
            
            # if args.lr_schedule != 'cyclic' and args.lr_schedule != 'fix' and args.lr_schedule != 'multistep' and args.lr_schedule != 'StepLR':
                
            lr = scheduler(epoch + (i + 1) / len(train_loader))
            # print(f'learning rate for epoch:{epoch} is :{lr}')
            # print("*"*80)
            opt.param_groups[0].update(lr=lr)
            
            if i == 0:
                first_batch = (X, y)

            if args.attack == 'fgsm':
                delta = attack_pgd_Alireza(model, X, y, epsilon, args.fgsm_alpha * epsilon, 1, 1, 'l_inf',fgsm_init=args.delta_init) 

            elif args.attack == 'zerograd':
                delta, full_delta = zero_grad(model, X, y, epsilon, args.fgsm_alpha * epsilon, args.zero_qval, args.zero_iters, args.delta_init)

            with torch.cuda.amp.autocast():
                output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                loss = criterion(output, y)
                logit_clean = model(normalize(X))
                
                # Forward pass with amp
                # with torch.cuda.amp.autocast():
                    ############ HAT #################
                if args.hat == True:
                    X_adv_hat = X + delta[:X.size(0)]
                    X_hr = X + 2 * (X_adv_hat - X)
                    y_hr = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                    out_help = model(normalize(torch.clamp(X_hr, min=lower_limit, max=upper_limit)))
                    loss_help = F.cross_entropy(out_help, y_hr, reduction='mean')
                else:
                    loss_help = 0.00
                    ##################################
                    
                    

                    
                    
                if args.kapa != 0.00:
                    # loss_robust = criterion_kl(F.log_softmax(output, dim=1),F.softmax(model(X), dim=1))
                    loss_robust = 0.00
                    logit_clean = model(normalize(X))
                    loss_clean = criterion(logit_clean, y)
                else:
                    loss_robust = torch.tensor([0.00]).cuda()
                    loss_clean = torch.tensor([0.00]).cuda()
                    # logit_clean = torch.tensor([0.00]).cuda()
                        
                    
                ########### CURE + TRADE ########################################
                # if epoch in regularizer_epochs:
                    ########### FMN_Linf for proper direction ##################
                    
                    # Total loss : loss + TRADE_loss + CURE_regulizer

                if args.delta == 'linf':
                    best_adv, r_linf = fmn(model=model, inputs=X , labels=y, norm = 2.0, steps=5)
                    r_linf = clamp(r_linf, -epsilon, epsilon)
                    best_adv = X+r_linf 
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
                    # regularizer, norm_grad = cure.regularizer(X, y, delta='FGSM', h=args.h, X_adv=normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                    
                    ############## Debug with full_delta ####################################################################################################
                    regularizer, norm_grad = cure.regularizer(model, X, y, delta='FGSM', h=args.h, X_adv=normalize(torch.clamp(X + full_delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                    curvature += regularizer.item()
                    gradient_norm += norm_grad.item()

                    # if curvature > 0.01000000000000:
                    #     print("Curvature exploding!")
                    #     print("*"*80)
                        # regularizer = args.lambda_*regularizer*500
                    
                    loss = loss + regularizer + gradient_norm
                    
                    # loss = loss + regularizer
                    
                elif args.delta == 'None':
                    loss = loss + loss_clean + (1/args.batch_size)*args.betta*loss_robust+args.gamma*loss_help
                
                else:
                    raise ValueError("Loss not computing")
                    
                    
                opt.zero_grad()           
                scaler.scale(loss).backward()
                # loss.backward()
                scaler.step(opt)
                # opt.step()
                
                scaler.update()

            train_loss += loss.item() * y.size(0)
            train_loss_clean+= loss_clean.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_acc_clean += (logit_clean.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            
            # if args.lr_schedule == 'cyclic':
            #     scheduler.step()
            if args.lr_schedule == 'multistep':
                scheduler.step()

            elif args.lr_schedule == 'StepLR':
                scheduler.step()
        
                
            
        # if args.early_stop:
            # Check current PGD robustness of model using random minibatch
        indices = torch.randperm(len(test_dataset))[:128]
        # Create the subset
        test_subset = Subset(test_dataset, indices)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
        test_loss_clean = 0
        test_loss_FGSM = 0
        test_acc = 0
        test_robust_loss = 0
        test_robust_acc = 0
        test_n = 0
        for i, (X, y) in enumerate(test_loader):
            
            X, y = X.cuda(), y.cuda()
            

            delta = attack_pgd_Alireza(model.eval(), X, y, epsilon, pgd_alpha, args.test_iters, args.test_restarts, 'l_inf', early_stop=args.eval, fgsm_init=None)
            delta = delta.detach()
            robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
            robust_loss = criterion(robust_output, y)

            output = model(normalize(X))
            loss = criterion(output, y)

            test_robust_loss += robust_loss.item() * y.size(0)
            test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            test_loss_clean += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            
            test_n += y.size(0)     
            print("="*60)
            print(f'Robust Accurcy for epoch:{epoch} is :{(test_robust_acc/test_n)*100}%')
            print("="*60)
            
            if test_robust_acc > best_test_robust_acc:
                torch.save(model.state_dict(),os.path.join(args.out_dir, f'model_best.pth'))
                best_test_robust_acc = test_robust_acc
        
        
            ############## Print Information ##########################################
            print(f'Curvature for epoch:{epoch} is :{curvature}')
            print("="*60)
            print(f'Gradient norm for epoch:{epoch} is :{gradient_norm}')
            print("="*60)
            print(f'Train Accuracy on FGSM samples for epoch:{epoch} is :{(train_acc/train_n)*100}%')
            print("="*60)
            print(f'Train Accuracy on clean samples for epoch:{epoch} is :{(train_acc_clean/train_n)*100}%')
            print("="*60)
            print(f'Train loss on clean samples for epoch:{epoch} is :{(train_loss_clean/train_n)}')
            print("="*60)
            print(f'Train loss for epoch:{epoch} is :{(train_loss/train_n)}')
            print("="*60)
            print(f'Test loss on clean samples for epoch:{epoch} is :{(test_loss_clean/test_n)}')
            print("="*60)
            print(f'Test Accuracy on clean samples for epoch:{epoch} is :{(test_acc/test_n)*100}%')
            epoch_time = time.time()
            print('Total epoch time: %.4f minutes', (epoch_time - start_epoch_time)/60)
            #######################################################################################################
            accuracy_df = accuracy_df.append({'epoch': epoch, 'loss_train_FGSM': train_loss/train_n ,'loss_train_clean': train_loss_clean/train_n,'ACC_train':(train_acc/train_n)*100,'ACC_Clean_train':(train_acc_clean/train_n)*100,'ACC_test_clean':(test_acc/test_n)*100,'Acc_test_PGD':(test_robust_acc/test_n)*100,'Curvature':curvature, 'grad_norm':gradient_norm,'test_loss_clean':test_loss_clean/test_n}, ignore_index=True)


    # Save the DataFrame to a CSV file
    accuracy_df.to_csv(results_csv, index=False)
    train_time = time.time()

    # # Evaluation
    # model_test = PreActResNet18().cuda()
    # # model_test = WideResNet().cuda()
    # # model_test = resnet(name='resnet18', num_classes=10).cuda()
    # model_test.load_state_dict(best_state_dict)
    # model_test.float()
    # model_test.eval()

    # # Select 1024 random indices from the test set
    # indices = torch.randperm(len(test_dataset))[:1024]
    # # Create the subset
    # test_subset = Subset(test_dataset, indices)
    # testloader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
    # pgd_loss, pgd_acc = evaluate_pgd(testloader, model_test, 20, 1) # zero-restarts
    # print('='*60)
    # print(f'pgd_loss is :{pgd_loss} and pgd_acc(robust acc) is :{pgd_acc*100}')
    # print('='*60)
    # # ACC_AA_PGD = evaluate_robust_accuracy_AA_APGD(model_test, testloader, 'cuda', epsilon=8/255)
    # # test_loss, test_acc = evaluate_standard(testloader, model_test)
    # # print(f'Robust accuray for AA-PGD is :{ACC_AA_PGD}')
    # # print('='*60)
    # # print(f'test_loss is :{test_loss} and clean_acc is:{test_acc*100}')
    




if __name__ == "__main__":
    main()
