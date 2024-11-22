import os
import os.path as osp
import cv2
import random

from .common import BaseDataset
from data.utils import np2Tensor

class Div2kTest(BaseDataset):
    def __init__(self, data_opt, **kwargs):
        super(Div2kTest, self).__init__(data_opt, **kwargs)

        gt_keys = sorted(os.listdir(self.deg_valid_dir))
        self.keys = sorted(list(set(gt_keys)))
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, item):
        gt_name = self.keys[item]
        gt_path = osp.join(self.deg_valid_dir, gt_name)

        gt = cv2.imread(gt_path)[..., ::-1]
        gts = np2Tensor(gt) # chw|rgb|uint8

        return {'gt': gts}
    
    def _get_patch(self, img, patch_size=48, scale=1):
        h, w = img.shape[:2] # hr
        tp = round(scale * patch_size)

        th = random.randint(0, (h-tp))
        tw = random.randint(0, (w-tp))

        return img[th: th+tp, tw: tw+tp, :]
    
    def get_patch(self, img):
        out = []
        img = self.augment(img)
        for _ in range(2):
            img_patch = self._get_patch(img, self.patch_size, self.scale)
            out.append(img_patch)
        return out
    
    def augment(self, img, flip=True, rot=True):
        hflip = flip and random.random() < 0.5
        vflip = flip and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        
        return img
