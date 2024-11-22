import random
import torch
import torch.nn as nn


def rgb2raw(img):
    R = img[:, 0, ...]
    G = img[:, 1, ...]
    B = img[:, 2, ...]

    raw = torch.zeros_like(R)
    h, w = raw.size()[1:]

    raw[:, 0:h:2, 0:w:2] = R[:, 0:h:2, 0:w:2]
    raw[:, 0:h:2, 1:w:2] = G[:, 0:h:2, 1:w:2]
    raw[:, 1:h:2, 0:w:2] = G[:, 1:h:2, 0:w:2]
    raw[:, 1:h:2, 1:w:2] = B[:, 1:h:2, 1:w:2]

    raw = raw * 65535.0 / 255.0

    return raw

def raw2rgb(raw):
    kernels = torch.tensor([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0],
                            [0.25, 0, 0.25], [0, 0, 0], [0.25, 0, 0.25],
                            [0, 0, 0], [0.5, 0, 0.5], [0, 0, 0],
                            [0, 0.5, 0], [0, 0, 0], [0, 0.5, 0],]).view(4, 1, 3, 3)
    
    rggb = torch.tensor([[4, 2], [3, 1], [0, 4], [4, 0], [1, 3], [2, 4]]).view(1, 3, 2, 2)

    b, h, w = raw.size()
    raw = raw.view(b, 1, h, w)
    raw_pad = torch.nn.functional.pad(raw, (1, 1, 1, 1), mode='reflect')
    c = torch.nn.functional.conv2d(raw_pad, kernels, stride=1)
    c = torch.cat((c, raw), dim=1)

    img = torch.gather(c, 1, rggb.repeat(1, 1, torch.div(h, 2, rounding_mode='floor'), torch.div(w, 2, rounding_mode='floor')).expand(b, -1, -1, -1))

    return img

def poisson_noise(lr_tensor, poisson_noise_level):
    input = torch.rand_like(lr_tensor) * poisson_noise_level
    return torch.poisson(input)

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
        self.p_noise = opt['poisson_noise']

    def __call__(self, lr_tensor, batch):
        if self.p_noise > 0.0:
            if lr_tensor.size()[1] == 3:
                lr_raw = rgb2raw(lr_tensor)
                lr_raw.add_(poisson_noise(lr_raw, self.p_noise))
                lr_tensor = raw2rgb(lr_raw)
            elif lr_tensor.size()[1] == 6:
                lr1 = lr_tensor[:, 0: 3, ...]
                lr_raw1 = rgb2raw(lr1)
                lr_raw1.add_(poisson_noise(lr_raw1, self.p_noise))
                lr1 = raw2rgb(lr_raw1)

                lr2 = lr_tensor[:, 3: 6, ...]
                lr_raw2 = rgb2raw(lr2)
                lr_raw2.add_(poisson_noise(lr_raw2, self.p_noise))
                lr2 = raw2rgb(lr_raw2)
                
                lr_tensor = torch.concat((lr1, lr2), dim=1)

        if self.g_noise > 0.0:
            norm_noise_level = torch.rand(batch, 1, 1, 1).to(self.device) if self.random else self.g_noise
            noise_level = norm_noise_level* self.g_noise
            noise = torch.randn_like(lr_tensor).mul_(noise_level).to(self.device)
            lr_tensor.add_(noise)

        return lr_tensor, norm_noise_level
