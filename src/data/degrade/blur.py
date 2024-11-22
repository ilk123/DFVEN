import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def cal_sigma(sig_x, sig_y, radians):
    sig_x = sig_x.view(-1, 1, 1)
    sig_y = sig_y.view(-1, 1, 1)
    radians = radians.view(-1, 1, 1)

    D = torch.cat([F.pad(sig_x ** 2, [0, 1, 0, 0]), 
                   F.pad(sig_y ** 2, [1, 0, 0, 0])], 1)
    U = torch.cat([torch.cat([radians.cos(), -radians.sin()], 2), 
                   torch.cat([radians.sin(), radians.cos(), 2])], 1)
    sigma = torch.bmm(U, torch.bmm(D, U.transpose(1, 2)))
    return sigma

def isotropic_gaussian_kernel(batch, kernel_size, sigma):
    ax = torch.arange(kernel_size).float() - kernel_size // 2

    xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
    yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)

    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma.view(-1, 1, 1) ** 2))

    return kernel / kernel.sum([1, 2], keepdim=True)

def anisotropic_gaussian_kernel(batch, kernel_size, covar):
    ax = torch.arange(kernel_size).float() - kernel_size // 2

    xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
    yy = ax.repeat(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
    xy = torch.stack([xx, yy], -1).view(batch, -1, 2)

    inverse_sigma = torch.inverse(covar)

    kernel = torch.exp(-0.5 * (torch.bmm(xy, inverse_sigma) * xy).sum(2)).view(batch, kernel_size, kernel_size)

    return kernel / kernel.sum([1, 2], keepdim=True)


class GaussianKernelTest(object):
    def __init__(self, device, type='isotropic') -> None:
        self.type = type
        self.device = device

        self.ksize = 7

        if self.type == 'isotropic':
            self.sigma = [0.2, 1.8, 2.8, 4.0]
        elif self.type == 'anisotropic':
            pass
        else:
            raise ValueError('Invalid Blur Type!')
    
    def __call__(self, batch):
        if self.type == 'isotropic':
            sigma = torch.Tensor([random.choice(self.sigma)])

            kernel = isotropic_gaussian_kernel(batch, self.ksize, sigma).to(self.device)

            return kernel, sigma.cuda()
        
        elif self.type == 'anisotropic':
            pass

class GaussianKernel(object):
    def __init__(self, device, opt, type='isotropic', sigma=2.0, theta=90, lambda1=0.2, lambda2=2.0, random=True):
        self.type = type
        self.random = random
        self.ksize = opt['blur_ksize']
        self.device = device

        if self.type == 'isotropic':
            self.sigma = opt.get('blur_sigma', sigma)
            self.sigma_min = opt['blur_sigma_min']
            self.sigma_max = opt['blur_sigma_max']

        elif self.type == 'anisotropic':
            self.theta = theta
            self.lambda1 = opt.get('blur_lambda1', lambda1)
            self.lambda2 = opt.get('blur_lambda2', lambda2)
            self.lambda_min = opt['blur_lambda_min']
            self.lambda_max = opt['blur_lambda_max']
        
        else:
            raise ValueError('Invalid Blur Type. ')
    
    def __call__(self, batch=4):
        if self.type == 'isotropic':
            if self.random:
                sigma_level = torch.rand(batch)
                sigma = sigma_level * (self.sigma_max - self.sigma_min) + self.sigma_min
            else:
                sigma = torch.ones(1) * self.sigma
            
            kernel = isotropic_gaussian_kernel(batch, self.ksize, sigma).to(self.device)

            return kernel, sigma_level.unsqueeze(0).to(self.device)
        
        elif self.type == 'anisotropic':
            if self.random:
                theta = torch.rand(batch) / 180 * math.pi

                lambda1 = torch.rand(batch) * (self.lambda_max - self.lambda_min) + self.lambda_min
                lambda2 = torch.rand(batch) * (self.lambda_max - self.lambda_min) + self.lambda_min
            else:
                theta = torch.ones(1) * self.theta / 180 * math.pi

                lambda1 = torch.ones(1) * self.lambda1
                lambda2 = torch.ones(1) * self.lambda2

            covar = cal_sigma(lambda1, lambda2, theta).to(self.device)

            kernel = anisotropic_gaussian_kernel(batch, self.ksize, covar).to(self.device)

            return kernel, covar


class BatchBlur(nn.Module):
    def __init__(self, kernel_size=21):
        super(BatchBlur, self).__init__()
        self.ksize = kernel_size

        if self.ksize % 2 == 1:
            self.pad = nn.ReflectionPad2d(kernel_size // 2)
        else:
            self.pad = nn.ReflectionPad2d((kernel_size // 2, kernel_size // 2 - 1, kernel_size // 2, kernel_size // 2 - 1))
    
    def forward(self, input, kernel):
        b, c, h, w = input.size()

        input_pad = self.pad(input)
        hp, wp = input_pad.size()[-2:]

        if len(kernel.size()) == 2:
            input_CBHW = input_pad.view((c*b, 1, hp, wp))

            kernel = kernel.contiguous().view((1, 1, self.ksize, self.ksize))

            return F.conv2d(input_CBHW, kernel, padding=0).view((b, c, h, w))
        else:
            input_CBHW = input_pad.view((1, c*b, hp, wp))

            kernel = kernel.contiguous().view((b, 1, self.ksize, self.ksize))
            kernel = kernel.repeat(1, c, 1, 1).view((b*c, 1, self.ksize, self.ksize))

            return F.conv2d(input_CBHW, kernel, groups=b*c).view((b, c, h, w))
