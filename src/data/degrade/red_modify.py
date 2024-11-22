import torch
import torch.nn as nn

import random
import numpy as np
from scipy import interpolate


class RedModifyTest(nn.Module):
    def __init__(self, device, rgb_range=255):
        super(RedModifyTest, self).__init__()
        self.device = device

        self.rgb_range = rgb_range

        self.xy = [[90, 160], [100, 150], [110, 140], [120, 130]]
    
    def _evaluate_polynomial(self, coeffs, x):
        n = coeffs.size(0) - 1
        result = torch.zeros_like(x)
        for i in range(n + 1):
            result += coeffs[i] * x**(n - i)
        return result
    
    def __call__(self, lr_tensor, batch):
        if lr_tensor.size()[1] == 3:
            lr_r = lr_tensor[:, 0, ...]
        elif lr_tensor.size()[1] % 3 == 0:
            lr_r = []
            for i in range(lr_tensor.size()[1] // 3):
                _lr_r = lr_tensor[:, 3*i, ...].unsqueeze(dim=1)
                lr_r.append(_lr_r)
            lr_r = torch.concat(lr_r, dim=1)
        
        xy = random.choice(self.xy)
        # x = np.array([xy[0]])
        # y = np.array([xy[1]])
        X = [0, xy[0], 255]
        Y = [0, xy[1], 255]
        poly = interpolate.lagrange(X, Y)
        
        coefficients = torch.from_numpy(np.array(poly))
        
        lr_r[0, ...] = self._evaluate_polynomial(coefficients, lr_r[0, ...])

        if lr_tensor.size()[1] == 3:
            lr_t = torch.concat((lr_r.unsqueeze(0), lr_tensor[:, 1: 3, ...]), dim=1)
        elif lr_tensor.size()[1] % 3 == 0:
            lr_t = torch.zeros_like(lr_tensor)
            for i in range(lr_tensor.size()[1] // 3):
                lr_t[:, 3*i, ...] = lr_r[:, i, ...].unsqueeze(dim=1)
                lr_t[:, 3*i+1: 3*i+3, ...] = lr_tensor[:, 3*i+1, 3*i+3, ...]
        # if lr_tensor.size()[1] == 6:
        #     lr_t = torch.concat((lr_r[:, 0, ...].unsqueeze(dim=1), lr_tensor[:, 1: 3, ...], lr_r[:, 1, ...].unsqueeze(dim=1), lr_tensor[:, 4: 6, ...]), dim=1)
        
        return lr_t, torch.Tensor([xy[0], xy[1]])


class RedModify(nn.Module):

    def __init__(self, device, opt, random, rgb_range=255):
        super(RedModify, self).__init__()
        self.device = device
        self.random = random

        self.rgb_range = rgb_range
        
        self.x_min = opt.get('red_x_min', 90)
        self.x_range = opt.get('red_x_range', 30)
        self.y_min = opt.get('red_y_min', 130)
        self.y_range = opt.get('red_y_range', 30)

    def _evaluate_polynomial(self, coeffs, x):
        n = coeffs.size(0) - 1
        result = torch.zeros_like(x)
        for i in range(n + 1):
            result += coeffs[i] * x**(n - i)
        return result
    
    def __call__(self, lr_tensor, batch):
        if lr_tensor.size()[1] == 3:
            lr_r = lr_tensor[:, 0, ...]
        elif lr_tensor.size()[1] % 3 == 0:
            lr_r = []
            for i in range(lr_tensor.size()[1] // 3):
                _lr_r = lr_tensor[:, 3*i, ...].unsqueeze(dim=1)
                lr_r.append(_lr_r)
            lr_r = torch.concat(lr_r, dim=1)
        
        x_level = np.random.rand(batch) if self.random else self.x_min
        y_level = np.random.rand(batch) if self.random else self.y_min
        x = x_level * self.x_range + self.x_min
        y = y_level * self.y_range + self.y_min
        poly = []
        for i in range(batch):
            X = [0, x[i], 255]
            Y = [0, y[i], 255]
            _poly = interpolate.lagrange(X, Y)
            poly.append(_poly.coefficients)
        
        coefficients = torch.from_numpy(np.array(poly))
        
        for i in range(batch):
            lr_r[i, ...] = self._evaluate_polynomial(coefficients[i], lr_r[i, ...])

        if lr_tensor.size()[1] == 6:
            lr_tensor = torch.concat((lr_r[:, 0, ...].unsqueeze(dim=1), lr_tensor[:, 1: 3, ...], lr_r[:, 1, ...].unsqueeze(dim=1), lr_tensor[:, 4: 6, ...]), dim=1)
        elif lr_tensor.size()[1] == 3:
            lr_tensor = torch.concat((lr_r.unsqueeze(dim=1), lr_tensor[:, 1: 3, ...]), dim=1)
        
        return lr_tensor, torch.stack([torch.from_numpy(y_level), torch.from_numpy(x_level)]).permute(1, 0).float().cuda()


