import torch
import torch.nn as nn
import numpy as np
import functools
import torch.nn.functional as F

def _get_patch(img, patch_size):
    n, t, _, th, tw = img.shape

    # # 每个batch内裁剪相同位置
    # tx = random.randrange(0, (tw-patch_size))
    # ty = random.randrange(0, (th-patch_size))
        
    # return img[:, :, ty: ty+patch_size, tx: tx+patch_size], [tx, ty]
    
    # 每个batch内裁剪不同位置，但每个时间序列内裁剪相同位置
    # 代码同下，但输入数据维度不同，此情况下输入为ntchw
    tx = np.random.randint(30, tw-30-patch_size, size=[img.shape[0]])
    ty = np.random.randint(30, th-30-patch_size, size=[img.shape[0]])
    out = torch.stack([img[i, :, :, ty[i]: ty[i]+patch_size, tx[i]: tx[i]+patch_size] for i in range(img.shape[0])], dim=0)

    # # 每张图像裁剪不同位置
    # # 此情况下输入为nchw
    # tx = np.random.randint(30, tw-30-patch_size, size=[img.shape[0]])
    # ty = np.random.randint(30, th-30-patch_size, size=[img.shape[0]])
    # out = torch.stack([img[i, :, ty[i]: ty[i]+patch_size, tx[i]: tx[i]+patch_size] for i in range(img.shape[0])], dim=0)
    
    return out, [tx, ty]

def get_patch(lr, patch_size=48):
    out = []
    for i in range(2):
        lr_patch, pos = _get_patch(lr, patch_size) # use the second image and pos for training
        out.append(lr_patch)
    out = torch.stack(out, dim=1).float()
    # out = np.array(out).transpose((1, 0, 2, 3, 4))
    # out_tensor = torch.from_numpy(out).float()
    return out, pos

def patch_clip(tensor, pos, s=2, psize=48, is_hr=False):
    """ Function of clip the original data to specified patch size
    
        Parameters:
            :tensor: a batch of original data (lr | gt | bi) in shape nchw
            :pos: the position of the cliped patch in lr data in shape 2n, list
            :scale: resolution upsample scale, int
            :patch_size: specified patch size for lr data, int 
            :is_hr: whether the tensor is high resolution, bool
    """
    tensor_patch = []
    if is_hr:
        for i in range(tensor.shape[0]):
            _tensor_patch = tensor[i, :, :, pos[1][i]*s: (pos[1][i]+psize)*s, pos[0][i]*s: (pos[0][i]+psize)*s]
            tensor_patch.append(_tensor_patch)
        tensor_patch = torch.stack(tensor_patch, 0)
    else:
        for i in range(tensor.shape[0]):
            _tensor_patch = tensor[i, :, :, pos[1][i]: pos[1][i]+psize, pos[0][i]: pos[0][i]+psize]
            tensor_patch.append(_tensor_patch)
        tensor_patch = torch.stack(tensor_patch, 0)
    
    return tensor_patch
    
def space_to_depth(x, scale=4):
    """ Equivalent to tf.space_to_depth()
    """

    n, c, in_h, in_w = x.size()
    out_h, out_w = in_h // scale, in_w // scale

    x_reshaped = x.reshape(n, c, out_h, scale, out_w, scale)
    x_reshaped = x_reshaped.permute(0, 3, 5, 1, 2, 4)
    output = x_reshaped.reshape(n, scale * scale * c, out_h, out_w)

    return output


def backward_warp(x, flow, mode='bilinear', padding_mode='border'):
    """ Backward warp `x` according to `flow`

        Both x and flow are pytorch tensor in shape `nchw` and `n2hw`

        Reference:
            https://github.com/sniklaus/pytorch-spynet/blob/master/run.py#L41
    """

    n, c, h, w = x.size()

    # create mesh grid
    iu = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(n, -1, h, -1)
    iv = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(n, -1, -1, w)
    grid = torch.cat([iu, iv], 1).to(flow.device)

    # normalize flow to [-1, 1]
    flow = torch.cat([
        flow[:, 0:1, ...] / ((w - 1.0) / 2.0),
        flow[:, 1:2, ...] / ((h - 1.0) / 2.0)], dim=1)

    # add flow to grid and reshape to nhw2
    grid = (grid + flow).permute(0, 2, 3, 1)

    # bilinear sampling
    # Note: `align_corners` is set to `True` by default in PyTorch version
    #        lower than 1.4.0
    if int(''.join(torch.__version__.split('.')[:2])) >= 14:
        output = F.grid_sample(
            x, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    else:
        output = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode)

    return output
