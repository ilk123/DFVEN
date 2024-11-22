import torch
import numpy as np
import torch.nn.functional as F

def _get_patch(img, patch_size):
    th, tw = img.shape[-2], img.shape[-1]
    
    tx = np.random.randint(30, tw-30-patch_size, size=[img.shape[0]])
    ty = np.random.randint(30, th-30-patch_size, size=[img.shape[0]])
    out = torch.stack([img[i, ..., ty[i]: ty[i]+patch_size, tx[i]: tx[i]+patch_size] for i in range(img.shape[0])], dim=0)
    
    return out, [tx, ty]

def get_patch(lr, patch_size=48):
    out = []
    for i in range(2):
        lr_patch, pos = _get_patch(lr, patch_size) # use the second image and pos for training
        out.append(lr_patch)
    out = torch.stack(out, dim=1).float()
    return out, pos

def patch_clip(tensor, pos, s=2, psize=48, is_hr=False):
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
