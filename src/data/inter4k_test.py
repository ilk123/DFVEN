import os
import os.path as osp
import numpy as np
import cv2
import torch

from .common import BaseDataset
from utils import retrieve_files


class Inter4kTest(BaseDataset):
    def __init__(self, data_opt, **kwargs):
        super(Inter4kTest, self).__init__(data_opt, **kwargs)
        
        if self.test_gt_dir is None:
            lr_keys = sorted(os.listdir(self.test_lr_dir))
            self.keys = sorted(list(set(lr_keys)))
        else:
            gt_keys = sorted(os.listdir(self.test_gt_dir))
            lr_keys = sorted(os.listdir(self.test_lr_dir))
            self.keys = sorted(list(set(gt_keys) & set(lr_keys)))

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        key = self.keys[index]

        if self.test_gt_dir is None:
            lr_seq = []
            gt_seq = []
            for frm_path in retrieve_files(osp.join(self.test_lr_dir, key)):
                frm = cv2.imread(frm_path)[..., ::-1]
                frm_gt = cv2.resize(frm, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                frm = frm.astype(np.float32) / 255.0
                lr_seq.append(frm)
                gt_seq.append(frm_gt)
            lr_seq = np.stack(lr_seq) # thwc|rgb|float32
            gt_seq = np.stack(gt_seq) # thwc|rgb|uint8

        else:
            lr_seq = []
            for frm_path in retrieve_files(osp.join(self.test_lr_dir, key)):
                frm = cv2.imread(frm_path)[..., ::-1].astype(np.float32) / 255.0
                lr_seq.append(frm)
            lr_seq = np.stack(lr_seq) # thwc|rgb|float32

            gt_seq = []
            for frm_path in retrieve_files(osp.join(self.test_gt_dir, key)):
                frm = cv2.imread(frm_path)[..., ::-1]
                gt_seq.append(frm)
            gt_seq = np.stack(gt_seq) # thwc|rgb|uint8

        gt_tsr = torch.from_numpy(np.ascontiguousarray(gt_seq)) # uint8
        lr_tsr = torch.from_numpy(np.ascontiguousarray(lr_seq)) # float32

        return {
            'gt': gt_tsr, 
            'lr': lr_tsr, 
            'seq_idx': key, 
            'frm_idx': sorted(os.listdir(osp.join(self.test_lr_dir, key)))
        }
