import random
import torch
import torch.nn as nn


class NoiseTest(nn.Module):
    def __init__(self, device):
        self.device = device

        self.n = [0, 10, 20, 30]
    
    def __call__(self, lr_tensor, batch):
        n = random.choice(self.n)
        noise = torch.randn_like(lr_tensor).mul_(n).to(self.device)
        lr_tensor.add_(noise)

        return lr_tensor, torch.Tensor([n])

class Noise(nn.Module):
    def __init__(self, device, opt, random=True):
        super(Noise, self).__init__()
        self.device = device
        self.random = random

        self.g_noise = opt['gaussian_noise']

    def __call__(self, lr_tensor, batch):
        if self.g_noise > 0.0:
            norm_noise_level = torch.rand(batch, 1, 1, 1).to(self.device) if self.random else self.g_noise
            noise_level = norm_noise_level* self.g_noise
            noise = torch.randn_like(lr_tensor).mul_(noise_level).to(self.device)
            lr_tensor.add_(noise)

        return lr_tensor, norm_noise_level
