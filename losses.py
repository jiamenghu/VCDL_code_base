
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

'''
scheduler
'''
class MyScheduler():
    def __init__(self,opts):
        self.best_psnr_gain = -100
        self.patience_max=opts['patience_max']
        self.patience=0
        self.optimizer=opts['optim']
        self.model=opts['model']
        self.lr_min=opts['lr_min']
        self.model_path=opts['model_path']
        self.lr_decay_rate=opts['lr_decay_rate']
        self.model_clone=None
        self.optim_clone=None
        self.lr=opts['lr']
        
    def step(self,cur_psnr):
        if cur_psnr>self.best_psnr_gain+0.001 :
            self.best_psnr_gain=cur_psnr
            self.patience=0
            self.model_clone = copy.deepcopy(self.model .state_dict()) 
            self.optim_clone = copy.deepcopy(self.optimizer.state_dict())
            torch.save(self.model.state_dict(), self.model_path)
        else:
            self.patience+=1
        if self.patience>self.patience_max:
            print('adjust lr:',self.lr, ' to ', self.lr*self.lr_decay_rate)
            self.model.load_state_dict(self.model_clone)
            self.optimizer.load_state_dict(self.optim_clone)
            self.lr*=self.lr_decay_rate
            self.adjust_learning_rate()
            self.patience=0

    def stop(self):
        if self.lr<self.lr_min: return True
        else :return False
    def adjust_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] =  self.lr 

'''
loss

'''

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

import torch_dct as dct_2d
class FrequencyMSELoss(nn.Module):

    def __init__(self,rate=1):
        super(FrequencyMSELoss, self).__init__()

        self.mse=nn.MSELoss()
        self.rate=rate
    def forward(self, x, y):
        diff=self.mse(x,y)

        fx=dct_2d.dct_2d(x)
        fy=dct_2d.dct_2d(y)

        Fdiff =self.mse(fx,fy)
        #print(diff,self.rate*Fdiff,diff+self.rate*Fdiff)
        # # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        # loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return diff+self.rate*Fdiff
        