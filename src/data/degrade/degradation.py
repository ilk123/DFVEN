import torch

from ..utils import tensor_f32_to_uint8
from .blur import GaussianKernel, GaussianKernelTest, BatchBlur
from .downsample import Bicubic
from .noise import Noise, NoiseTest
from .low_light import LowLight, LowLightTest
from .red_modify import RedModify, RedModifyTest


class DataPrepare(object):
    def __init__(self, device, opt, data_type='degradation'):
        self.device = device
        self.data_type = data_type
        self.random = opt['random']

        self.scale = opt['scale']

        self.bicubic = Bicubic()
        self.b_kernel = GaussianKernel(self.device, opt, random=self.random)
        self.blur = BatchBlur(opt['blur_ksize'])
        self.light = LowLight(self.device, opt, random=self.random)
        self.noise = Noise(self.device, opt, random=self.random)
        self.r_modify = RedModify(self.device, opt, random=self.random)
    
    def __call__(self, data):
        if self.data_type == 'blur':
            gt_data = data['gt']
            b, n, c, H, W = gt_data.size()
            b_kernels, sigma = self.b_kernel(b)

            gt_blured = self.blur(gt_data.contiguous().view(b, -1, H, W), b_kernels)
            gt_blured_multied = tensor_f32_to_uint8(gt_blured)
            lr_data = self.bicubic(gt_blured_multied, scale=1/self.scale)

            lr_data = torch.clamp(lr_data, 0, 255).mul_(1/255)
            lr_data = lr_data.view(b, n, c, H//int(self.scale), W//int(self.scale))
        
            return {'gt': gt_data, 'lr': lr_data, 'para': sigma}
        
        elif self.data_type == 'noise':
            gt_data = data['gt']
            gt_data_multied = tensor_f32_to_uint8(gt_data)
            b, n, c, H, W = gt_data_multied.size()

            lr_data = self.bicubic(gt_data_multied.contiguous().view(b, -1, H, W), scale=1/self.scale)
            lr_data, noise = self.noise(lr_data, b)

            lr_data = torch.clamp(lr_data, 0, 255).mul_(1/255)
            lr_data = lr_data.view(b, n, c, H//int(self.scale), W//int(self.scale))
        
            return {'gt': gt_data, 'lr': lr_data, 'para': noise}
        
        elif self.data_type == 'light':
            gt_data = data['gt']
            gt_data_multied = tensor_f32_to_uint8(gt_data)
            b, n, c, H, W = gt_data_multied.size()

            gt_lowLighted, gamma = self.light(gt_data_multied.contiguous().view(b, -1, H, W), b)
            lr_data = self.bicubic(gt_lowLighted, scale=1/self.scale)

            lr_data = torch.clamp(lr_data, 0, 255).mul_(1/255)
            lr_data = lr_data.view(b, n, c, H//int(self.scale), W//int(self.scale))
        
            return {'gt': gt_data, 'lr': lr_data, 'para': gamma}
        
        elif self.data_type == 'red':
            gt_data = data['gt']
            gt_data_multied = tensor_f32_to_uint8(gt_data)
            b, n, c, H, W = gt_data_multied.size()
            
            lr_data = self.bicubic(gt_data_multied.contiguous().view(b, -1, H, W), scale=1/self.scale)
            lr_data, ratio = self.r_modify(lr_data, b)

            lr_data = torch.clamp(lr_data, 0, 255).mul_(1/255)
            lr_data = lr_data.view(b, n, c, H//int(self.scale), W//int(self.scale))
        
            return {'gt': gt_data, 'lr': lr_data, 'para': ratio}
        
        elif self.data_type == 'all':
            gt_data = data['gt']
            b, n, c, H, W = gt_data.size()
            b_kernels, sigma = self.b_kernel(b)

            gt_blured = self.blur(gt_data.contiguous().view(b, -1, H, W), b_kernels)
            gt_blured_multied = tensor_f32_to_uint8(gt_blured)
            gt_lowLighted, _ = self.light(gt_blured_multied, b)
            lr_lowLighted = self.bicubic(gt_lowLighted, scale = 1/self.scale)
            lr_noised, _ = self.noise(lr_lowLighted, b)
            lr_data, _ = self.r_modify(lr_noised, b)

            lr_data = torch.clamp(lr_data, 0, 255).mul_(1/255)
            lr_data = lr_data.view(b, n, c, H//int(self.scale), W//int(self.scale))
        
            return {'gt': gt_data, 'lr': lr_data, 'para': sigma}

        else:
            raise ValueError('Unrecognized degradation type: {}'.format(self.data_type))


class DataPrepareTest(object):
    def __init__(self, device, opt, data_type='degradation'):
        self.device = device
        self.data_type = data_type
        
        self.scale = opt['scale']

        self.bicubic = Bicubic()
        self.b_kernel = GaussianKernelTest(self.device)
        self.blur = BatchBlur(opt['blur_ksize'])
        self.r_modify = RedModifyTest(self.device)
        self.noise = NoiseTest(self.device)
        self.light = LowLightTest(self.device)
    
    def __call__(self, data):
        if self.data_type == 'blur':
            gt_data = data['gt']
            gt_data_multied = gt_data.mul(255)
            b, n, c, H, W = gt_data_multied.size()
            b_kernels, sigma = self.b_kernel(b)

            gt_blured = self.blur(gt_data.contiguous().view(b, -1, H, W), b_kernels)
            lr_data = self.bicubic(gt_blured, scale=1/self.scale)

            lr_data = torch.clamp(lr_data, 0, 255).mul_(1/255)
            lr_data = lr_data.view(b, n, c, H // int(self.scale), W // int(self.scale))
        
            return {'gt': gt_data, 'lr': lr_data, 'para': sigma}
        
        elif self.data_type == 'noise':
            gt_data = data['gt']
            gt_data_multied = gt_data.mul(255)
            b, n, c, H, W = gt_data_multied.size()
            
            lr_data = self.bicubic(gt_data.contiguous().view(b, -1, H, W), scale=1/self.scale)
            lr_data, noise = self.noise(lr_data, b)

            lr_data = torch.clamp(lr_data, 0, 255).mul_(1/255)
            lr_data = lr_data.view(b, n, c, H // int(self.scale), W // int(self.scale))
        
            return {'gt': gt_data, 'lr': lr_data, 'para': noise}
        
        elif self.data_type == 'light':
            gt_data = data['gt']
            gt_data_multied = gt_data.mul(255)
            b, n, c, H, W = gt_data_multied.size()

            gt_lowLighted, gamma = self.light(gt_data.contiguous().view(b, -1, H, W), b)
            lr_data = self.bicubic(gt_lowLighted, scale=1/self.scale)

            lr_data = torch.clamp(lr_data, 0, 255).mul_(1/255)
            lr_data = lr_data.view(b, n, c, H // int(self.scale), W // int(self.scale))
        
            return {'gt': gt_data, 'lr': lr_data, 'para': gamma}
        
        elif self.data_type == 'red':
            gt_data = data['gt']
            gt_data_multied = gt_data.mul(255)
            b, n, c, H, W = gt_data_multied.size()
            
            lr_data = self.bicubic(gt_data.contiguous().view(b, -1, H, W), scale=1/self.scale)
            lr_data, ratio = self.r_modify(lr_data, b)

            lr_data = torch.clamp(lr_data, 0, 255).mul_(1/255)
            lr_data = lr_data.view(b, n, c, H // int(self.scale), W // int(self.scale))
        
            return {'gt': gt_data, 'lr': lr_data, 'para': ratio}
        
        elif self.data_type == 'all':
            gt_data = data['gt']
            gt_data_multied = gt_data.mul(255).contiguous()
            b, t, c, H, W = gt_data_multied.size()
            b_kernels, sigma = self.b_kernel(b)

            gt_blured = self.blur(gt_data_multied.view(b, -1, H, W), b_kernels)
            gt_lowLighted, _ = self.light(gt_blured, b)
            lr_lowLighted = self.bicubic(gt_lowLighted, scale = 1/self.scale)
            lr_noised, _ = self.noise(lr_lowLighted, b)
            lr_data, _ = self.r_modify(lr_noised, b)

            lr_data = torch.clamp(lr_data, 0, 255).mul_(1/255)
            lr_data = lr_data.view(b, t, c, H // int(self.scale), W // int(self.scale))
        
            return {'gt': gt_data, 'lr': lr_data, 'para': sigma}
