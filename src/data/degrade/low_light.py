import random
import torch
import torch.nn as nn


def linear_transform(img, alpha):
    return alpha * img

def gamma_transform(img, gamma):
    img.mul_(1/255)
    return torch.pow(img, gamma).mul_(255)

class LowLightTest(nn.Module):
    def __init__(self, device):
        super(LowLightTest, self).__init__()
        self.device = device

        self.abg = [[0.9, 0.5, 2.5], [0.9, 0.5, 2.0], [0.9, 0.5, 1.5], [0.9, 0.5, 1.0]]
    
    def __call__(self, hr_tensor, batch):
        abg = random.choice(self.abg)
        hr_tensor = linear_transform(hr_tensor, abg[0])
        hr_tensor = gamma_transform(hr_tensor, abg[2])
        hr_tensor = linear_transform(hr_tensor, abg[1])

        return hr_tensor, torch.Tensor([abg[2]])


class LowLight(nn.Module):
    def __init__(self, device, opt, alpha=0.9, beta=0.8, gamma=1.5, random=True):
        super(LowLight, self).__init__()
        self.device = device
        self.random = random

        self.alpha = opt.get('light_alpha', alpha)
        self.alpha_min = opt['light_alpha_min']
        self.alpha_max = opt['light_alpha_max']

        self.beta = opt.get('light_beta', beta)
        self.beta_min = opt['light_beta_min']
        self.beta_max = opt['light_beta_max']

        self.gamma = opt.get('light_gamma', gamma)
        self.gamma_min = opt['light_gamma_min']
        self.gamma_max = opt['light_gamma_max']

    def __call__(self, hr_tensor, batch):
        if self.random:
            alpha_level = torch.rand(batch, 1, 1, 1).to(self.device)
            beta_level = torch.rand(batch, 1, 1, 1).to(self.device)
            gamma_level = torch.rand(batch, 1, 1, 1).to(self.device)
            alpha = alpha_level * (self.alpha_max - self.alpha_min) + self.alpha_min
            beta = beta_level * (self.beta_max - self.beta_min) + self.beta_min
            gamma = gamma_level * (self.gamma_max - self.gamma_min) + self.gamma_min

            hr_tensor = linear_transform(hr_tensor, alpha)
            hr_tensor = gamma_transform(hr_tensor, gamma)
            hr_tensor = linear_transform(hr_tensor, beta)
        
        else:
            hr_tensor = linear_transform(hr_tensor, self.alpha)
            hr_tensor = gamma_transform(hr_tensor, self.gamma)
            hr_tensor = linear_transform(hr_tensor, self.beta)

        return hr_tensor, torch.stack([alpha_level, beta_level, gamma_level], dim=1)
