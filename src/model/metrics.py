import torch
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

from data.utils import rgb_to_ycbcr
from .LPIPS.models.dist_model import DistModel


def compute_PSNR(true_img_cur, pred_img_cur, psnr_colorspace):
    if psnr_colorspace == 'rgb':
        true_img = true_img_cur
        pred_img = pred_img_cur
    elif psnr_colorspace == 'y':
        # convert to ycbcr, and keep the y channel
        true_img = rgb_to_ycbcr(true_img_cur)[..., 0]
        pred_img = rgb_to_ycbcr(pred_img_cur)[..., 0]
    else:
        raise ValueError('Unrecognized colorspace type: {}'.format(psnr_colorspace))

    diff = true_img.astype(np.float64) - pred_img.astype(np.float64)
    RMSE = np.sqrt(np.mean(np.power(diff, 2)))

    if RMSE == 0:
        return np.inf

    PSNR = 20 * np.log10(255.0 / RMSE)
    return PSNR

def compute_SSIM(true_img_cur, pred_img_cur, ssim_colorspace):
    if ssim_colorspace == 'rgb':
        true_img = true_img_cur
        pred_img = pred_img_cur
    elif ssim_colorspace == 'y':
        # convert to ycbcr, and keep the y channel
        true_img = rgb_to_ycbcr(true_img_cur)[..., 0]
        pred_img = rgb_to_ycbcr(pred_img_cur)[..., 0]
    else:
        raise ValueError('Unrecognized colorspace type: {}'.format(ssim_colorspace))
    
    SSIM = ssim(true_img, pred_img)
    return SSIM

def compute_LPIPS(img1, img2):
    # to tensor
    img1 = torch.FloatTensor(img1).unsqueeze(0).permute(0, 3, 1, 2)
    img2 = torch.FloatTensor(img2).unsqueeze(0).permute(0, 3, 1, 2)

    # normalize to [-1, 1]
    img1 = img1 * 2.0 / 255.0 - 1.0
    img2 = img2 * 2.0 / 255.0 - 1.0

    with torch.no_grad():
        dm = DistModel()
        dm.initialize(
                    model='net-lin',
                    net='alex',
                    colorspace='rgb',
                    spatial=False,
                    use_gpu=True,
                    gpu_ids=[0],
                    version=0.1)
        LPIPS = dm.forward(img1, img2)
    
    return LPIPS

def compute_tOF(true_img_cur, pred_img_cur, true_img_pre, pred_img_pre):
    true_img_cur = cv2.cvtColor(true_img_cur, cv2.COLOR_RGB2GRAY)
    pred_img_cur = cv2.cvtColor(pred_img_cur, cv2.COLOR_RGB2GRAY)
    true_img_pre = cv2.cvtColor(true_img_pre, cv2.COLOR_RGB2GRAY)
    pred_img_pre = cv2.cvtColor(pred_img_pre, cv2.COLOR_RGB2GRAY)

    # forward flow
    true_OF = cv2.calcOpticalFlowFarneback(
        true_img_pre, true_img_cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    pred_OF = cv2.calcOpticalFlowFarneback(
        pred_img_pre, pred_img_cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # EPE
    diff_OF = true_OF - pred_OF
    tOF = np.mean(np.sqrt(np.sum(diff_OF**2, axis=-1)))

    return tOF

def compute_tLP(true_img_cur, pred_img_cur, true_img_pre, pred_img_pre):
    true_img_cur = np.ascontiguousarray(true_img_cur)
    pred_img_cur = np.ascontiguousarray(pred_img_cur)
    true_img_pre = np.ascontiguousarray(true_img_pre)
    pred_img_pre = np.ascontiguousarray(pred_img_pre)

    # forward LPIPS
    true_LP = compute_LPIPS(true_img_pre, true_img_cur)
    pred_LP = compute_LPIPS(pred_img_pre, pred_img_cur)

    # EPE
    diff_LP = (true_LP - pred_LP).cpu().numpy()
    tLP = np.mean(np.sqrt(np.sum(diff_LP**2, axis=-1)))

    return tLP
