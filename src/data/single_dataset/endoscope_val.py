import os
import os.path as osp
import numpy as np
import cv2
import torch

from .common import BaseDataset
from utils import retrieve_files

class EndoscopeVal(BaseDataset):
    def __init__(self, data_opt, **kwargs):
        super(EndoscopeVal, self).__init__(data_opt, **kwargs)

        gt_keys = sorted(os.listdir(self.valid_gt_dir))
        lr_keys = sorted(os.listdir(self.valid_lr_dir))
        self.keys = sorted(list(set(gt_keys) & set(lr_keys)))

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        key = self.keys[index]

        gt_seq = []
        for frm_path in retrieve_files(osp.join(self.valid_gt_dir, key)):
            frm = cv2.imread(frm_path)[..., ::-1]
            gt_seq.append(frm)
        gt_seq = np.stack(gt_seq) # thwc|rgb|uint8

        lr_seq = []
        for frm_path in retrieve_files(osp.join(self.valid_lr_dir, key)):
            frm = cv2.imread(frm_path)[..., ::-1].astype(np.float32) / 255.0
            lr_seq.append(frm)
        lr_seq = np.stack(lr_seq) # thwc|rgb|float32

        gt_tsr = torch.from_numpy(np.ascontiguousarray(gt_seq)) # uint8
        lr_tsr = torch.from_numpy(np.ascontiguousarray(lr_seq)) # float32

        return {
            'gt': gt_tsr, 
            'lr': lr_tsr, 
            'seq_idx': key, 
            'frm_idx': sorted(os.listdir(osp.join(self.valid_gt_dir, key)))
        }
