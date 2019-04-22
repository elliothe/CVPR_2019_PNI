import torch
import torch.nn as nn

class Normalize_layer(nn.Module):
    
    def __init__(self, mean, std):
        super(Normalize_layer, self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1), requires_grad=False)
        
    def forward(self, input):
        
        return input.sub(self.mean).div(self.std)


class noise_Normalize_layer(nn.Module):
    
    def __init__(self, mean, std, input_noise=False):
        super(noise_Normalize_layer, self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.input_noise = input_noise
        self.alpha_i = nn.Parameter(torch.Tensor([0.25]), requires_grad = True)
        
    def forward(self, input):
        output = input.sub(self.mean).div(self.std)
        
        input_std = output.std().item()
        input_noise = output.clone().normal_(0, input_std)
        
        return output + input_noise*self.alpha_i*self.input_noise
