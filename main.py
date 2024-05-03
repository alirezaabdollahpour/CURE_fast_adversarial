import argparse
import copy
import logging
import os
import time
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


from models import *
from utils import *
from cure import *

import warnings
warnings.filterwarnings('ignore')



logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--h', default=1.5, type=float, help='hyperparameter for CURE regulizer')
    parser.add_argument('--lambda_', default=0.5, type=float, help='weight for CURE regulizer')
    parser.add_argument('--betta', default=1.0, type=float, help='weight for TRADE loss')
    parser.add_argument('--data-dir', default='cifar-data', type=str)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0.0, type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=10, type=float, help='Step size')
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


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    pgd_alpha = (2 / 255.) / std

    model = PreActResNet18().cuda()
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)

    # Necessary for FP16
    scaler = torch.cuda.amp.GradScaler()
    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=True)
    criterion = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction='sum')
    cure = CURE(model, opt=opt)

    if args.delta_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()

    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # Training
    prev_robust_acc = 0.
    start_train_time = time.time()
    for epoch in range(args.epochs):
        print("Start training")
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        curvature = 0.0
        for i, (X, y) in tqdm(enumerate(train_loader)):
            X, y = X.cuda(), y.cuda()
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
                output = model(X + delta[:X.size(0)]) # gradient for theta to train with SGD or Adam
                loss = criterion(output, y)
                loss_robust = criterion_kl(F.log_softmax(output, dim=1),F.softmax(model(X), dim=1))     
                
            ########### CURE ########################################
            regularizer, grad_norm = cure.regularizer(X, y, h=args.h)
            curvature += regularizer.item()
            # Total loss : loss + TRADE_loss + CURE_regulizer
            loss = loss + (1/args.batch_size)*(args.lambda_)*regularizer + (1.0 / args.batch_size)*args.betta*loss_robust
            # loss = loss + args.betta*loss_robust
            opt.zero_grad()
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()
        if args.early_stop:
            # Check current PGD robustness of model using random minibatch
            X, y = first_batch
            pgd_delta = attack_pgd(model, X, y, epsilon, pgd_alpha, 5, 1, opt)
            with torch.no_grad():
                output = model(clamp(X + pgd_delta[:X.size(0)], lower_limit, upper_limit))
            robust_acc = (output.max(1)[1] == y).sum().item() / y.size(0)
            print("="*40)
            print(f'Robust Accurcy for epoch:{epoch} is :{robust_acc*100}%')
            print("="*40)
            if robust_acc - prev_robust_acc < -0.2:
                break
            prev_robust_acc = robust_acc
            best_state_dict = copy.deepcopy(model.state_dict())
        epoch_time = time.time()
        ############## Print Information ##########################################
        print(f'curvature for epoch:{epoch} is :{curvature}')
        print("="*40)
        print(f'Train Accuracy for epoch:{epoch} is :{(train_acc/train_n)*100}%')
        print("="*40)
        epoch_time = time.time()
        print('Total epoch time: %.4f minutes', (epoch_time - start_epoch_time)/60)
        ###########################################################################

        lr = scheduler.get_lr()[0]


    train_time = time.time()
    if not args.early_stop:
        best_state_dict = model.state_dict()
    torch.save(best_state_dict, os.path.join(args.out_dir, 'model.pth'))

    # Evaluation
    model_test = PreActResNet18().cuda()
    model_test.load_state_dict(best_state_dict)
    model_test.float()
    model_test.eval()

    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 20, 10)
    test_loss, test_acc = evaluate_standard(test_loader, model_test)




if __name__ == "__main__":
    main()