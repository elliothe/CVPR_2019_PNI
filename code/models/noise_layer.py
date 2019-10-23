import torch.nn as nn
import math
import torch.nn.functional as F
import torch
import numpy as np


class noise_Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, pni='channelwise', w_noise=True):
        super(noise_Linear, self).__init__(in_features, out_features, bias)
        
        self.pni = pni
        if self.pni is 'layerwise':
            self.alpha_w = nn.Parameter(torch.Tensor([0.25]), requires_grad = True)
        elif self.pni is 'channelwise':
            self.alpha_w = nn.Parameter(torch.ones(self.out_features).view(-1,1)*0.25,
                                        requires_grad=True)
        elif self.pni is 'elementwise':
            self.alpha_w = nn.Parameter(torch.ones(self.weight.size())*0.25, requires_grad = True)
        
        self.w_noise = w_noise

    def forward(self, input):
        
        with torch.no_grad():
            std = self.weight.std().item()
            noise = self.weight.clone().normal_(0,std)

        noise_weight = self.weight + self.alpha_w * noise * self.w_noise
        output = F.linear(input, noise_weight, self.bias)
        
        return output 



class noise_Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, pni='channelwise', w_noise=True):
        super(noise_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                         padding, dilation, groups, bias)

        self.pni = pni
        if self.pni is 'layerwise':
            self.alpha_w = nn.Parameter(torch.Tensor([0.25]), requires_grad = True)
        elif self.pni is 'channelwise':
            self.alpha_w = nn.Parameter(torch.ones(self.out_channels).view(-1,1,1,1)*0.25,
                                        requires_grad = True)     
        elif self.pni is 'elementwise':
            self.alpha_w = nn.Parameter(torch.ones(self.weight.size())*0.25, requires_grad = True)  
        
        self.w_noise = w_noise    


    def forward(self, input):

        with torch.no_grad():
            std = self.weight.std().item()
            noise = self.weight.clone().normal_(0,std)

        noise_weight = self.weight + self.alpha_w * noise * self.w_noise
        output = F.conv2d(input, noise_weight, self.bias, self.stride, self.padding, self.dilation,
                        self.groups)

        return output
                      