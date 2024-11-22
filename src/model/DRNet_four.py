import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools

from .common import SRNet_Single
from .DNet import DnetWithoutTail
from .utils import get_patch, patch_clip, backward_warp, space_to_depth


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
        lr_patches, patch_pos = get_patch(lr, self.patch_size) # n2crr, n2

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

        hr = self.SRnet(lr_query, degrade_fea) # ncRR

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