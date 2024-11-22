import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .DNet import DnetWithoutTail
from .LiteFlowNet.LFNet import LFNet
from .utils import get_patch, patch_clip


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
    

class SRNet(nn.Module):
    def __init__(self, nf=64, nf2=16, nb=5):
        super(SRNet, self).__init__()

        self.nb = nb

        self.compress = nn.Sequential(
            nn.Linear(512, 64, bias=False), 
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


class Dfnet(nn.Module):
    def __init__(self, opt, **kwargs):
        super(Dfnet, self).__init__()
        for kw, args in opt.items():
            setattr(self, kw, args)
        
        for kw, args in kwargs.items():
            setattr(self, kw, args)
        
        self.FNet = LFNet()
        self.load_state_dict({ 'FNet.' + strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-liteflownet/network-default.pytorch', file_name='liteflownet-default').items() })
        for param in self.FNet.parameters():
            param.requires_grad = False

        self.conv_in = nn.Sequential(
            nn.Conv2d(self.gen_in_nc, self.gen_nf, 3, 1, 1, bias=True), 
            nn.LeakyReLU(0.1, True)
        )

        self.prev_compress = nn.Sequential(
            nn.Conv2d(self.gen_in_nc * 2 + 2, self.gen_nf, 3, 1, 1, bias=True), 
            nn.LeakyReLU(0.1, True)
        )

        self.Dnet = DnetWithoutTail(opt, deg_dim=self.deg_dim)
        self.SRnet = SRNet(self.gen_nf, self.gen_nf2, self.gen_nb)

        module_misr = [
            ResNetBlock(self.gen_nf, 3, 1, 1, bias=True) for i in range(self.gen_nr)
        ]
        module_misr.append(nn.Conv2d(self.gen_nf, self.gen_nf2, 3, 1, 1, bias=True))
        self.misr = nn.Sequential(*module_misr)

        module_err_res = [
            ResNetBlock(self.gen_nf2, 3, 1, 1, bias=True) for i in range(self.gen_nr)
        ]
        module_err_res.append(nn.Conv2d(self.gen_nf2, self.gen_nf2, 3, 1, 1, bias=True))
        self.err_res = nn.Sequential(*module_err_res)

        module_decode = [
            ResNetBlock(self.gen_nf2, 3, 1, 1, bias=True) for i in range(self.gen_nr)
        ]
        module_decode.append(nn.Conv2d(self.gen_nf2, self.gen_nf, 3, 1, 1, bias=True))
        self.decode = nn.Sequential(*module_decode)

        if self.scale == 2:
            self.upsample = nn.Sequential(
                nn.Conv2d(self.gen_nf2 * (self.tempo_range - 1), 4 * self.gen_out_nc, 3, 1, 1, bias=True), 
                nn.Dropout(p=0.1), 
                nn.Conv2d(4 * self.gen_out_nc, 4 * self.gen_out_nc, 3, 1, 1, bias=True),
                nn.PixelShuffle(2)
            )
        elif self.scale == 4:
            self.upsample = nn.Sequential(
                nn.Conv2d(self.gen_nf2 * (self.tempo_range - 1), 16 * self.gen_out_nc, 3, 1, 1, bias=True),
                nn.Dropout(p=0.1),
                nn.Conv2d(16 * self.gen_out_nc, 16 * self.gen_out_nc, 3, 1, 1, bias=True),
                nn.PixelShuffle(2),
                nn.Conv2d(4 * self.gen_out_nc, 4 * self.gen_out_nc, 3, 1, 1, bias=True),
                nn.PixelShuffle(2)
            )

    def generate_dummy_input(self, lr_size, is_seq=False):
        if is_seq:
            t, c, h, w = lr_size
            lr = torch.rand(1, t, c, h, w, dtype=torch.float32)

            return lr
        else:
            c, lr_h, lr_w = lr_size
            s = self.scale

            lr_curr = torch.rand(1, c, lr_h, lr_w, dtype=torch.float32)
            lr_prev = torch.rand(1, c, lr_h, lr_w, dtype=torch.float32)
            hr_prev = torch.rand(1, c, s * lr_h, s * lr_w, dtype=torch.float32)

            data_dict = {
                'lr_curr': lr_curr,
                'lr_prev': lr_prev,
                'hr_prev': hr_prev, 
                'is_train': False
            }
            return data_dict
    
    def calculate_flow(self, img1, img2):
        '''
            img1: shape of nchw
            img2: shape of nchw
        '''
        
        assert(img1.shape[2] == img2.shape[2])
        assert(img1.shape[3] == img2.shape[3])

        W = img1.shape[3]
        H = img2.shape[2]

        W_ = int(math.floor(math.ceil(W / 32.0) * 32.0))
        H_ = int(math.floor(math.ceil(H / 32.0) * 32.0))

        img1_ = F.interpolate(input=img1, size=(H_, W_), mode='bilinear', align_corners=False)
        img2_ = F.interpolate(input=img2, size=(H_, W_), mode='bilinear', align_corners=False)

        flow = F.interpolate(input=self.FNet(img1_, img2_), size=(H, W), mode='bilinear', align_corners=False)

        flow[:, 0, :, :] *= float(W) / float(W_)
        flow[:, 1, :, :] *= float(H) / float(H_)

        return flow
    
    def forward(self, lr):
        n, t, c, h, w = lr.size()
        p = self.patch_size
        lr = lr.flip(dims=[1])
        x = lr[:, 0, ...]
        neigbor = lr[:, 1:, ...]

        flow_frame = []
        # calculate optical flows
        for i in range(t - 1):
            with torch.no_grad():
                flow_frame.append(self.calculate_flow(x, neigbor[:, i, ...]))
        
        # generate image patches
        p_frames, patch_pos = get_patch(lr, p)
        query_patch = p_frames[:, 1, 0, ...]
        key_patch = p_frames[:, 0, 0, ...]
        neigbor_patch = p_frames[:, 1, 1:, ...]

        # clip optical flows
        f_frame = []
        for i in range(t - 1):
            flow_patch = patch_clip(flow_frame[i].unsqueeze(dim=1), patch_pos, self.scale, self.patch_size, is_hr=False).squeeze(dim=1)
            f_frame.append(self.prev_compress(torch.concat((query_patch, neigbor_patch[:, i, ...], flow_patch), dim=1)))

        # calculate degradation representations
        degrade_fea, logits, labels = self.Dnet(query_patch, key_patch, is_train=True)
        
        res = F.interpolate(query_patch, scale_factor=self.scale, mode='bilinear', align_corners=False)
        query_patch_in = self.conv_in(query_patch)

        R_frame = []
        for j in range(t - 1):
            r0 = self.SRnet(query_patch_in, degrade_fea)

            r1 = self.misr(f_frame[j])

            err = self.err_res(r0 - r1)
            r = r0 + err
            R_frame.append(r)
            query_patch_in = self.decode(r)
        
        out = torch.concat(R_frame, dim=1)
        res += F.tanh(self.upsample(out))

        return {
            'hr': res, 
            'logits': logits, 
            'labels': labels, 
            'patch_pos': patch_pos
        }
    
    def infer(self, lr):
        n, t, c, h, w = lr.size()
        p = self.patch_size
        lr = lr.flip(dims=[1])
        x = lr[:, 0, ...]
        neigbor = lr[:, 1:, ...]

        # calculate optical flows
        f_frame = []
        for i in range(t - 1):
            with torch.no_grad():
                flow_frame = self.calculate_flow(x, neigbor[:, i, ...])
            f_frame.append(self.prev_compress(torch.concat((x, neigbor[:, i, ...], flow_frame), dim=1)))

        degrade_fea = self.Dnet(x, x, is_train=False)
        
        res = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        x_in = self.conv_in(x)

        R_frame = []
        for j in range(t - 1):
            r0 = self.SRnet(x_in, degrade_fea)
            r1 = self.misr(f_frame[j])

            err = self.err_res(r0 - r1)
            r = r0 + err
            R_frame.append(r)
            x_in = self.decode(r)
        
        out = torch.concat(R_frame, dim=1)
        res += F.tanh(self.upsample(out))

        return res
