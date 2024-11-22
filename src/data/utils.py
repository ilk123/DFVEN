import torch
import numpy as np

def pad_sequence(lr_data, padding_mode='reflect', n_pad_front=0):
    """
    Parameters:
        :param lr_data: tensor in shape tchw
    """

    if padding_mode == 'reflect':
        lr_data = torch.cat(
            [lr_data[1: 1 + n_pad_front, ...].flip(0), lr_data], dim=0)

    elif padding_mode == 'replicate':
        lr_data = torch.cat(
            [lr_data[:1, ...].expand(n_pad_front, -1, -1, -1), lr_data],
            dim=0)

    elif padding_mode == 'dual-reflect':
        lr_data = torch.cat(
            [lr_data[1: 1+n_pad_front, ...].flip(0), lr_data, lr_data[-1-n_pad_front: -1, ...].flip(0)],
            dim=0)

    else:
        raise ValueError('Unrecognized padding mode: {}'.format(
            padding_mode))

    return lr_data, n_pad_front

def float32_to_uint8(inputs):
    """ Convert np.float32 array to np.uint8

        Parameters:
            :param input: np.float32, (NT)CHW, [0, 1]
            :return: np.uint8, (NT)CHW, [0, 255]
    """
    return np.uint8(np.clip(np.round(inputs * 255), 0, 255))

def tensor_f32_to_uint8(inputs):
    """ Convert torch.float32 array to torch.uint8

        Parameters:
            :param input: torch.float32, NCHW, [0, 1]
            :return: torch.uint8, NCHW, [0, 255]
    """
    return (inputs * 255).clamp(0, 255).to(torch.uint8)

def rgb_to_ycbcr(img):
    """ Coefficients are taken from the  official codes of DUF-VSR
        This conversion is also the same as that in BasicSR

        Parameters:
            :param  img: rgb image in type np.uint8
            :return: ycbcr image in type np.uint8
    """

    T = np.array([
        [0.256788235294118, -0.148223529411765,  0.439215686274510],
        [0.504129411764706, -0.290992156862745, -0.367788235294118],
        [0.097905882352941,  0.439215686274510, -0.071427450980392],
    ], dtype=np.float64)

    O = np.array([16, 128, 128], dtype=np.float64)

    img = img.astype(np.float64)
    res = np.matmul(img, T) + O
    res = res.clip(0, 255).round().astype(np.uint8)

    return res

def np2Tensor(data, rgb_range=1):
    np_transpose = np.ascontiguousarray(data.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float()
    tensor.mul_(rgb_range / 255)

    return tensor
