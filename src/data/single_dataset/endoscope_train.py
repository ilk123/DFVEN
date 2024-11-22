import os.path as osp
import random
import pickle
import torch
import numpy as np

from .common import BaseDataset

class EndoscopeTrain(BaseDataset):
    def __init__(self, data_opt, **kwargs):
        super(EndoscopeTrain, self).__init__(data_opt, **kwargs)

        meta = pickle.load(open(osp.join(self.train_data_dir, 'meta_info.pkl'), 'rb'))
        self.keys = sorted(meta['keys'])

        self.env = self.init_lmdb(self.train_data_dir)
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        key = self.keys[index]
        idx, (n, h, w), frms = self.parse_lmdb_key(key)
        c = 3 if self.train_data_type.lower() == 'rgb' else 1

        # if self.moving_first_frame and (random.uniform(0, 1) > self.moving_factor):
        #     frm = self.read_lmdb_frame(self.env, key, size=(h, w, c))
        #     frm = frm.transpose(2, 0, 1) # chw|rgb|uint8

        #     offsets = np.floor(
        #         np.random.uniform(-3.5, 4.5, size=(self.tempo_range, 2)))
        #     offsets = offsets.astype(np.int32)
        #     pos = np.cumsum(offsets, axis=0)
        #     min_pos = np.min(pos, axis=0)
        #     topleft_pos = pos - min_pos
        #     range_pos = np.max(pos, axis=0) - min_pos
        #     c_h, c_w = h - range_pos[0], w - range_pos[1]

        #     for i in range(self.tempo_range):
        #         top, left = topleft_pos[i]
        #         frms.append(frm[:, top: top+c_h, left: left+c_w].copy())
        # else:           
        #     frms = self.read_lmdb_frame(self.env, key, (n, h, w, c))
        #     frms = frms.transpose(0, 3, 1, 2)
        frms = self.read_lmdb_frame(self.env, key, (n, h, w, c))
        frms = frms.transpose(0, 3, 1, 2).squeeze()
        
        pats = self.augment_sequence(frms)
        tsr = torch.FloatTensor(np.ascontiguousarray(pats)) / 255.0
        return {'gt': tsr}
    
    def augment_sequence(self, pats):
        # flip spatially
        axis = random.randint(1, 2)
        if axis > 1:
            pats = np.flip(pats, axis)
        
        # # flip temporally
        # axis = random.randint(0, 1)
        # if axis < 1:
        #     pats = np.flip(pats, axis)
        
        # rotate
        k = random.randint(0, 3)
        pats = np.rot90(pats, k, (1, 2))

        return pats
