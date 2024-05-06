from utils import *
import torch
import torch.nn as nn

class LLR():
    
    def __init__(self, net, opt, epsilon,lambda_):

        self.scaler = torch.cuda.amp.GradScaler()
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.opt = opt
        self.epsilon = epsilon
        self.lambda_ = lambda_
    def get_uniform_delta(self, input, eps, requires_grad=True):
        delta = torch.zeros(input.shape).cuda()
        delta.uniform_(-eps, eps)
        delta.requires_grad = requires_grad
        return delta
    
    def get_input_grad(self, net, X, y, opt, eps, delta_init='none', backprop=False):
        if delta_init == 'none':
            delta = torch.zeros_like(X, requires_grad=True)
        elif delta_init == 'random_uniform':
            delta = self.get_uniform_delta(X.shape, eps, requires_grad=True)

        with torch.cuda.amp.autocast():
            output = self.net(X + delta)
            loss = self.criterion(output, y)
            grad = torch.autograd.grad(loss, delta, create_graph=True if backprop else False)[0]
        if not backprop:
            grad, delta = grad.detach(), delta.detach()
        return grad


    def regularizer(self, inputs, targets):

        '''
        Regularizer term in LLR
        '''
        
        x_2 = inputs + self.epsilon*(2*torch.rand(inputs.shape, device = inputs.device) - 1)
        x_3 = inputs + self.epsilon*(2*torch.rand(inputs.shape, device = inputs.device) - 1)
        
        g = self.get_input_grad(self.net, x_2, targets, self.opt, self.epsilon, delta_init='none', backprop=True)
        with torch.cuda.amp.autocast():
            out = self.net(torch.cat((x_2,x_3),dim=0))
            criterion = nn.CrossEntropyLoss(reduction='none')
        
            lin_err = self.mse(criterion(out[bs:], targets), criterion(out[:bs], targets) + ((x_3-x_2)*g).sum(dim=[1,2,3]))

        return lin_err





