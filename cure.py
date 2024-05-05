from utils import *
import torch
import torch.nn as nn

class CURE():
    
    def __init__(self, net, opt):

        self.scaler = torch.cuda.amp.GradScaler()
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        self.opt = opt
    
    def get_uniform_delta(self, input, eps, requires_grad=True):
        delta = torch.zeros(input.shape).cuda()
        delta.uniform_(-eps, eps)
        delta.requires_grad = requires_grad
        return delta

    def _find_z(self, inputs, targets, h):
        '''
        Finding the direction in the regularizer
        '''
        inputs.requires_grad_(True)
        outputs = self.net.eval()(inputs)
        loss_z = self.criterion(self.net.eval()(inputs), targets)
        
        with torch.cuda.amp.autocast():
            self.scaler.scale(loss_z).backward()
            
        grad = inputs.grad.data + 0.0
        norm_grad = grad.norm().item()
        z = torch.sign(grad).detach() + 0.
        z = 1.*(h) * (z+1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None]+1e-7)

        inputs.detach()
        self.net.zero_grad()

        return z, norm_grad


    def regularizer(self, inputs, targets, delta, h):

        '''
        Regularizer term in CURE
        '''
        if delta == 'classic':
            z, norm_grad = self._find_z(inputs, targets, h)
            inputs.requires_grad_(True)
            with torch.cuda.amp.autocast():
                outputs_pos = self.net.eval()(inputs + z)
                outputs_orig = self.net.eval()(inputs)

                loss_pos = self.criterion(outputs_pos, targets)
                loss_orig = self.criterion(outputs_orig, targets)
                
            grad_diff = torch.autograd.grad((loss_pos-loss_orig), inputs, only_inputs=True)[0]

            reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1)
            # reg = (1./h**2)*(reg**2)
            reg = (reg**2)
            self.net.zero_grad()

            return torch.sum(reg), norm_grad
        
        elif delta == 'random':
            
            z = self.get_uniform_delta(inputs, eps=8/255, requires_grad=True)
            # This part is good for AT with FGSM such that instead buliding z, you can use direction of FGSM or others proper direciotns in training by passing delta!
            inputs.requires_grad_(True)
            z.requires_grad_(True)
            with torch.cuda.amp.autocast():
                outputs_pos = self.net.eval()(inputs + z)
                outputs_orig = self.net.eval()(inputs)

                loss_pos = self.criterion(outputs_pos, targets)
                loss_orig = self.criterion(outputs_orig, targets)
                
            grad_diff = torch.autograd.grad((loss_pos-loss_orig), inputs, only_inputs=True)[0]

            reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1)
            # reg = (1./h**2)*(reg**2)
            reg = (reg**2)
            self.net.zero_grad()

            # return torch.sum(reg) / float(inputs.size(0))
            return torch.sum(reg)

        elif delta == 'linf':
            
            inputs.requires_grad_(True)
            with torch.cuda.amp.autocast():
                outputs_pos = self.net.eval()(inputs + delta[:inputs.size(0)])
                outputs_orig = self.net.eval()(inputs)

                loss_pos = self.criterion(outputs_pos, targets)
                loss_orig = self.criterion(outputs_orig, targets)
                
            grad_diff = torch.autograd.grad((loss_pos-loss_orig), inputs, only_inputs=True)[0]

            reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1)
            # reg = (1./h**2)*(reg**2)
            reg = (reg**2)
            self.net.zero_grad()

            return torch.sum(reg)

