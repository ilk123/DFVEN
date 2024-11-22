import cv2
import torch
import numpy as np
import torch.nn as nn
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
        
        # xy = random.choice(self.xy)
        # x = np.array([xy[:, 0]])
        # y = np.array([xy[:, 1]])
        poly = []
        for i in range(batch):
            X = [0, self.xy[i][0], 255]
            Y = [0, self.xy[i][1], 255]
            _poly = interpolate.lagrange(X, Y)
            poly.append(_poly.coefficients)
        
            coefficients = torch.from_numpy(np.array(poly))
        
        # for i in range(batch):
            lr_r[i, ...] = self._evaluate_polynomial(coefficients[i], lr_r[i, ...])

        if lr_tensor.size()[1] == 6:
            lr_tensor = torch.concat((lr_r[:, 0, ...].unsqueeze(dim=1), lr_tensor[:, 1: 3, ...], lr_r[:, 1, ...].unsqueeze(dim=1), lr_tensor[:, 4: 6, ...]), dim=1)
        elif lr_tensor.size()[1] == 3:
            lr_tensor = torch.concat((lr_r.unsqueeze(1), lr_tensor[:, 1: 3, ...]), dim=1)
        
        return lr_tensor


img = 'M:/XHD/imageSR/data/DIV2K/DIV2K_valid_HR/0801.png'

tensor = (torch.from_numpy(cv2.imread(img)) / 255).permute(2, 0, 1)
tensor = torch.stack([tensor, tensor, tensor, tensor]).cuda()

red = RedModifyTest(device=torch.device('cuda:0'))

output = red(tensor, 4)

imgs = output.permute(0, 2, 3, 1).cpu().numpy()
imgs = np.clip(imgs * 255, 0, 255).astype(np.uint8)

print(imgs[0] == img[1])

for i in range(4):
    img = imgs[i]
    cv2.imshow('1', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
