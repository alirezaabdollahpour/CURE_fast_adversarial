
import time
import shutil
import sys

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

from autoattack import AutoAttack
# from robustbench.robustbench.utils import load_model
# import robustbench




def normalize(X):
    return (X - mu)/std






cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
# CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
# CIFAR100_STD = (0.2673, 0.2564, 0.2762)

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)


class Normalize:
    def __init__(self, mu, std):
        self.mu = mu
        self.std = std

    def __call__(self, data):
        return (data - self.mu) / self.std




def str_to_bool(v):
    """Convert string to bool."""
    return bool(int(v))

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def get_loaders(dir_, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    num_workers = 2
    train_dataset = datasets.CIFAR10(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(
        dir_, train=False, transform=test_transform, download=True)
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


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None):
    # print(f'epsilon is :{epsilon}')
    scaler = torch.cuda.amp.GradScaler()
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            with torch.cuda.amp.autocast():
                output = model(X + delta)
                index = torch.where(output.max(1)[1] == y)
                if len(index[0]) == 0:
                    break
                loss = F.cross_entropy(output, y)
            if opt is not None:
                scaler.scale(loss).backward()
                # scaler.update()
                ############## amp ###########################
                # with amp.scale_loss(loss, opt) as scaled_loss:
                    # scaled_loss.backward()
                ##############################################
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts):
    epsilon = (8 / 255.) / std
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return pgd_loss/n, pgd_acc/n


def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n







def evaluate_robust_accuracy_AA_Complete(model, data_loader, device, epsilon):
    
    # Put the model in evaluation mode
    model.eval()

    # epsilon = epsilon / 255.
    adversary = AutoAttack(model, norm='Linf', eps=0.031, version='standard',verbose=False)

    robust_accuracy = 0.0
    total = 0

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Run the attack
        inputs_adv = adversary.run_standard_evaluation(normalize(inputs), labels)
        with torch.no_grad():
            outputs = model(normalize(normalize(torch.clamp(inputs_adv, min=lower_limit, max=upper_limit))))
            _, predicted = torch.max(outputs, 1)
            robust_accuracy += (predicted == labels).sum().item()
        total += labels.size(0)

    # Calculate final robust accuracy
    robust_accuracy = robust_accuracy / total
    print(f'Robust accuracy under attack: {robust_accuracy * 100:.2f}%')
    return robust_accuracy






def interpolate_models(model1, model2, alpha=0.5):
    print("start interpolation")
    """
    Interpolates between the state_dicts of two models based on a coefficient, loads
    the result into the second model, and returns it.

    Parameters:
        model1 (torch.nn.Module): First model whose state_dict is used as the base.
        model2 (torch.nn.Module): Second model where the interpolated state_dict is loaded.
        alpha (float): Interpolation coefficient.

    Returns:
        torch.nn.Module: The second model with the interpolated state_dict loaded.
    """
    # Extract the state_dicts from both models
    sd1 = model1.state_dict()
    sd2 = model2.state_dict()
    
    # Check if models are compatible
    if sd1.keys() != sd2.keys():
        raise ValueError("Models have different architectures and cannot be interpolated")
    
    # Create a new state_dict to store the interpolated parameters
    interpolated_sd = {}
    
    # Interpolate each parameter
    with torch.cuda.amp.autocast():
        for key in sd1:
            if sd1[key].size() != sd2[key].size():
                raise ValueError("Parameter sizes do not match for the models.")
            # Perform the interpolation
            interpolated_sd[key] = (1 - alpha) * sd1[key] + alpha * sd2[key]
    
    # Load the interpolated state_dict into model2
    model1.load_state_dict(interpolated_sd)
    
    # model1 = model1.train()
    return model1