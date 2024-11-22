import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BaseEncoder(nn.Module):
    def __init__(self, in_nc):
        super(BaseEncoder, self).__init__()

        self.resnet = models.resnet101()

        self.features = None
        self.resnet.avgpool.register_forward_hook(self.hook_fn)

        self.resnet.layer4._modules['2'] = nn.Sequential(
            nn.Conv2d(2048, 512, (1, 1), (1, 1), bias=False), 
            nn.BatchNorm2d(512), 
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=False), 
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True)
        )

        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 512), 
            nn.ReLU(True),
            nn.Linear(512, 256)
        )

        self.tail = nn.Sequential(
            nn.Linear(256, 64), 
            nn.ReLU(True), 
            nn.Linear(64, 1), 
            nn.Sigmoid()
        )

    def hook_fn(self, module, input, output):
        self.features = output

    def forward(self, x):

        out = self.resnet(x)
        fea = self.features.squeeze(-1).squeeze(-1)
        res = self.tail(out).squeeze(-1).squeeze(-1)

        return fea, out, res



class BaseEncoderWithoutTail(nn.Module):
    def __init__(self, in_nc):
        super(BaseEncoderWithoutTail, self).__init__()

        self.resnet = models.resnet101()

        self.features = None
        self.resnet.avgpool.register_forward_hook(self.hook_fn)

        self.resnet.layer4._modules['2'] = nn.Sequential(
            nn.Conv2d(2048, 512, (1, 1), (1, 1), bias=False), 
            nn.BatchNorm2d(512), 
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=False), 
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True)
        )

        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 512), 
            nn.ReLU(True),
            nn.Linear(512, 256)
        )

    def hook_fn(self, module, input, output):
        self.features = output

    def forward(self, x):

        out = self.resnet(x)
        fea = self.features.squeeze(-1).squeeze(-1)

        return fea, out

class DA_conv(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(DA_conv, self).__init__()
        self.channels_out = out_nc
        self.channels_in = in_nc
        self.kernel_size = 3

        self.kernel = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 64 * self.kernel_size * self.kernel_size, bias=False)
        )
        self.conv = nn.Conv2d(in_nc, out_nc, 1, 1, 0, bias=True)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        b, c, h, w = x[0].size()

        kernel = self.kernel(x[1]).view(-1, 1, self.kernel_size, self.kernel_size)
        out = self.relu(F.conv2d(x[0].view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2))
        out = self.conv(out.view(b, -1, h, w))

        return out


class DABlock(nn.Module):
    def __init__(self, nf):
        super(DABlock, self).__init__()

        self.da_conv = DA_conv(nf, nf)
        self.conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''

        out = self.relu(self.da_conv(x))
        out = self.conv(out) + x[0]

        return out
    

class SRNet_Single(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=5, scale=2):
        super(SRNet_Single, self).__init__()

        self.s = scale
        self.nb = nb

        self.compress = nn.Sequential(
            nn.Linear(2048, 64, bias=False), 
            nn.LeakyReLU(0.1, True)
        )

        self.conv_in = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True), 
            nn.ReLU(inplace=True)
        )

        modules_body = [
            DABlock(nf) for _ in range(self.nb)
        ]
        modules_body.append(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
        self.body = nn.Sequential(*modules_body)

        self.c1 = nn.Conv2d(nf, 4 * out_nc, 3, 1, 1, bias=True)
        self.c2 = nn.PixelShuffle(2)

    def forward(self, x, degrade):
        degrade = self.compress(degrade)

        out = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv_in(x)
        res = x
        for i in range(self.nb):
            res = self.body[i]([res, degrade])
        res = self.body[-1](res)
        res += x

        res = self.c1(res)
        res = self.c2(res)
        out += F.sigmoid(res)

        return out
    

class SRNet(nn.Module):
    def __init__(self, nf=64, nf2=16, nb=5):
        super(SRNet, self).__init__()

        self.nb = nb

        self.compress = nn.Sequential(
            nn.Linear(2048, 64, bias=False), 
            nn.LeakyReLU(0.1, True)
        )

        modules_body = [
            DABlock(nf) for _ in range(self.nb)
        ]
        modules_body.append(nn.Conv2d(nf, nf2, 3, 1, 1, bias=True))
        self.body = nn.Sequential(*modules_body)

    def forward(self, fx, degrade):
        degrade = self.compress(degrade)

        res = fx
        for i in range(self.nb):
            res = self.body[i]([res, degrade])
        res = self.body[-1](res)

        return res


class ResNetBlock(nn.Module):
    def __init__(self, nf, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResNetBlock, self).__init__()
        self.c1 = nn.Conv2d(nf, nf, kernel_size, stride, padding, bias=bias)
        self.r1 = nn.LeakyReLU(0.1, False)
        self.c2 = nn.Conv2d(nf, nf, kernel_size, stride, padding, bias=bias)
        self.r2 = nn.LeakyReLU(0.1, False)

    def forward(self, x):
        out = self.r1(self.c1(x))
        out += x
        out = self.r2(out)
        return out
