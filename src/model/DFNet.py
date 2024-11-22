import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .common import SRNet, ResNetBlock
from .DNet import DnetWithoutTail
from .LiteFlowNet.LFNet import LFNet
from .utils import get_patch, patch_clip


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

        self.Dnet = DnetWithoutTail(opt, deg_dim=self.deg_dim//self.deg_num)
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
        # 计算流场
        for i in range(t - 1):
            with torch.no_grad():
                flow_frame.append(self.calculate_flow(x, neigbor[:, i, ...]))
        
        # 生成图像块
        p_frames, patch_pos = get_patch(lr, p) # nt2crr, n2
        query_patch = p_frames[:, 1, 0, ...]
        key_patch = p_frames[:, 0, 0, ...]
        neigbor_patch = p_frames[:, 1, 1:, ...]

        # 计算帧间初始特征张量
        f_frame = []
        for i in range(t - 1):
            flow_patch = patch_clip(flow_frame[i].unsqueeze(dim=1), patch_pos, self.scale, self.patch_size, is_hr=False).squeeze(dim=1)
            f_frame.append(self.prev_compress(torch.concat((query_patch, neigbor_patch[:, i, ...], flow_patch), dim=1)))

        # 计算退化特征向量
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

        # 计算流场和帧间初始特征张量
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
