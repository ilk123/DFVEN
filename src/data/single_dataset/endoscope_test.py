import os
import os.path as osp
import numpy as np
import cv2
import torch

from .common import BaseDataset


class EndoscopeTest(BaseDataset):
    def __init__(self, data_opt, **kwargs):
        super(EndoscopeTest, self).__init__(data_opt, **kwargs)
        
        if self.test_gt_dir is None:
            lr_keys = sorted(os.listdir(self.test_lr_dir))
            self.keys = sorted(list(set(lr_keys)))
        elif self.test_lr_dir is None:
            gt_keys = sorted(os.listdir(self.test_gt_dir))
            self.keys = sorted(list(set(gt_keys)))
        else:
            gt_keys = sorted(os.listdir(self.test_gt_dir))
            lr_keys = sorted(os.listdir(self.test_lr_dir))
            self.keys = sorted(list(set(gt_keys) & set(lr_keys)))

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        key = self.keys[index]
        lr_img = cv2.imread(osp.join(self.test_lr_dir, key))[..., ::-1].astype(np.float32) / 255.0
        gt_img = cv2.imread(osp.join(self.test_gt_dir, key))[..., ::-1]

        try:
            gt_tsr = torch.from_numpy(np.ascontiguousarray(gt_img)).unsqueeze(dim=0) # uint8
            lr_tsr = torch.from_numpy(np.ascontiguousarray(lr_img)).unsqueeze(dim=0) # float32
        except:
            gt_tsr = torch.from_numpy(np.ascontiguousarray(gt_img)).unsqueeze(dim=0) # uint8
            lr_tsr = torch.empty_like(gt_tsr)

        return {
            'gt': gt_tsr, 
            'lr': lr_tsr, 
            'img_idx': key, 
        }
