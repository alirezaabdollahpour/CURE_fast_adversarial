
import torch
import torch.nn as nn



class CURE():
    
    def __init__(self, net, opt, lambda_):

        self.scaler = torch.cuda.amp.GradScaler()
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        # self.opt = opt
        self.eps = 8./255.
        self.lambda_ = lambda_
        self.input = None
        self.input_adv = None
    
    def train(self):
        self.net.train()  # Set the model to train mode
    
    def eval(self):
        self.net.eval()  # Set the model to eval mode

    
    def get_uniform_delta(self, input, eps, requires_grad=True):
        delta = torch.zeros(input.shape).cuda()
        delta.uniform_(-eps, eps)
        delta.requires_grad = requires_grad
        return delta

    def get_input_grad(self, net, X, y, opt, eps, delta_init='none', backprop=False):
        if delta_init == 'none':
            delta = torch.zeros_like(X, requires_grad=True)
            # delta = X
        elif delta_init == 'random_uniform':
            delta = self.get_uniform_delta(X.shape, eps, requires_grad=True)
        
        elif delta_init == 'fmn':
            delta = self.input_adv - self.input
            delta.requires_grad_(True)
        
        else:
            raise ValueError("Invalid value for delta_init. Expected 'fmn', got: {}".format(delta_init))
            
            
                        
        with torch.cuda.amp.autocast():
            # self.net.eval()
            output = net(X + delta)
            loss = self.criterion(output, y)
            grad = torch.autograd.grad(loss, delta, create_graph=True if backprop else False)[0]
        if not backprop:
            grad, delta = grad.detach(), delta.detach()
            
        return grad

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


    def regularizer(self,net, inputs,targets,opt, delta, h, X_adv=None):

        '''
        Regularizer term in CURE
        '''
        self.input = inputs
        self.input_adv = X_adv
        self.opt = opt
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
            # reg = (reg**2)
            self.net.zero_grad()
            return self.lambda_*torch.sum(reg)
        
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
            # reg = (reg**2)
            self.net.zero_grad()

            # return torch.sum(reg) / float(inputs.size(0))
            return self.lambda_*torch.sum(reg)

        elif delta == 'linf' and X_adv != None:
            
            g_2 = self.get_input_grad(net, inputs, targets, self.opt, self.eps, delta_init='none', backprop=False)
            g_3 = self.get_input_grad(net, X_adv, targets, self.opt, self.eps, delta_init='none', backprop=True)
            
            
            reg = ((g_2-g_3)*(g_2-g_3)).mean(dim=0).sum()
            
            g_2_norm = (g_2*g_2).mean(dim=0).sum()
            g_3_norm = (g_3*g_3).mean(dim=0).sum()
            
            grad_norm = g_2_norm + g_3_norm
                        
            return self.lambda_*reg, self.lambda_*grad_norm
        

        elif delta == 'FGSM' and X_adv != None:
            

            g_2 = self.get_input_grad(net, inputs, targets, self.opt, self.eps, delta_init='none', backprop=False)
            g_3 = self.get_input_grad(net, X_adv, targets, self.opt, self.eps, delta_init='none', backprop=True)
            
            
            reg = ((g_2-g_3)*(g_2-g_3)).mean(dim=0).sum()
            
            g_2_norm = (g_2*g_2).mean(dim=0).sum()
            g_3_norm = (g_3*g_3).mean(dim=0).sum()
            
            grad_norm = g_2_norm + g_3_norm
                        
            return self.lambda_*reg, self.lambda_*grad_norm
            # return self.lambda_*reg, self.lambda_*g_2_norm
            # return self.lambda_*reg
            # return reg

