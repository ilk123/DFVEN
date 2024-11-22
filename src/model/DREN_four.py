import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

from .DFVEN import DABlock
from .DNet import DnetWithoutTail
from .utils import get_patch


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


class DrnetFour(nn.Module):
    def __init__(self, opt, **kwargs):
        super(DrnetFour, self).__init__()
        for kw, args in opt.items():
            setattr(self, kw, args)
        
        for kw, args in kwargs.items():
            setattr(self, kw, args)

        self.Dnet1 = DnetWithoutTail(opt, deg_dim=self.deg_dim//self.deg_num)
        self.Dnet2 = DnetWithoutTail(opt, deg_dim=self.deg_dim//self.deg_num)
        self.Dnet3 = DnetWithoutTail(opt, deg_dim=self.deg_dim//self.deg_num)
        self.Dnet4 = DnetWithoutTail(opt, deg_dim=self.deg_dim//self.deg_num)
        self.SRnet = SRNet_Single(self.gen_in_nc, self.gen_out_nc, self.gen_nf, self.gen_nb, self.scale)

        self.upsample_func = functools.partial(
            F.interpolate, scale_factor=self.scale, mode='bilinear',
            align_corners=False)

    def generate_dummy_input(self, lr_size):
        c, lr_h, lr_w = lr_size
        s = self.scale

        lr = torch.rand(1, c, lr_h, lr_w, dtype=torch.float32)

        data_dict = {
            'lr': lr,
            'is_train': False
        }
        return data_dict
    
    def forward_single(self, lr, is_train=True):
        '''Parameters:
            lr: the current lr data in shape nchw
        '''

        # lr patches generate and save patch position
        lr_patches, patch_pos = get_patch(lr, self.patch_size)

        # estimate degardation representation on one patch by DNet
        lr_query = lr_patches[:, 1, ...]
        lr_key = lr_patches[:, 0, ...]
        degrade_fea1, logits1, labels1 = self.Dnet1(lr_query, lr_key, is_train=True)  # n(b//4)
        degrade_fea2, logits2, labels2 = self.Dnet2(lr_query, lr_key, is_train=True)  # n(b//4)
        degrade_fea3, logits3, labels3 = self.Dnet3(lr_query, lr_key, is_train=True)  # n(b//4)
        degrade_fea4, logits4, labels4 = self.Dnet4(lr_query, lr_key, is_train=True)  # n(b//4)

        degrade_fea = torch.concat((degrade_fea1, degrade_fea2, degrade_fea3, degrade_fea4), dim=1) # nb
        logits = [logits1, logits2, logits3, logits4]
        labels = [labels1, labels2, labels3, labels4]

        hr = self.SRnet(lr_query, degrade_fea)

        return {
            'hr': hr, 
            'logits': logits, 
            'labels': labels, 
            'patch_pos': patch_pos
        }
    def infer_single(self, lr, is_train=False):
        '''Parameters:
            lr: the current lr data in shape nchw\
        '''

        # estimate degardation representation by DNet
        degrade_fea1 = self.Dnet1(lr, lr, is_train)  # n(b//4)
        degrade_fea2 = self.Dnet2(lr, lr, is_train)  # n(b//4)
        degrade_fea3 = self.Dnet3(lr, lr, is_train)  # n(b//4)
        degrade_fea4 = self.Dnet4(lr, lr, is_train)  # n(b//4)
        degrade_fea = torch.concat((degrade_fea1, degrade_fea2, degrade_fea3, degrade_fea4), dim=1) # nb

        hr = self.SRnet(lr, degrade_fea) # ncHW

        return hr