from utils import *
import torch
import torch.nn as nn

class CURE():
    
    def __init__(self, net, opt):

        self.scaler = torch.cuda.amp.GradScaler()
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        self.opt = opt

    def _find_z(self, inputs, targets, h):
        '''
        Finding the direction in the regularizer
        '''
        inputs.requires_grad_(True)
        outputs = self.net.eval()(inputs)
        loss_z = self.criterion(self.net.eval()(inputs), targets)
        # with amp.scale_loss(loss_z, self.opt) as scaled_loss:
        #     # loss_z.backward()
        #     scaled_loss.backward()
        with torch.cuda.amp.autocast():
            self.scaler.scale(loss_z).backward()
            
        grad = inputs.grad.data + 0.0
        norm_grad = grad.norm().item()
        z = torch.sign(grad).detach() + 0.
        z = 1.*(h) * (z+1e-5) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None]+1e-5)

        inputs.detach()
        self.net.zero_grad()

        return z, norm_grad


    def regularizer(self, inputs, targets, h):

        '''
        Regularizer term in CURE
        '''
        z, norm_grad = self._find_z(inputs, targets, h)

        inputs.requires_grad_(True)
        with torch.cuda.amp.autocast():
            outputs_pos = self.net.eval()(inputs + z)
            outputs_orig = self.net.eval()(inputs)

            loss_pos = self.criterion(outputs_pos, targets)
            loss_orig = self.criterion(outputs_orig, targets)
            
        grad_diff = torch.autograd.grad((loss_pos-loss_orig), inputs, only_inputs=True)[0]

        reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1)
        reg = (1./h**2)*(reg**2)
        self.net.zero_grad()

        return torch.sum(reg), norm_grad


