import os.path as osp
import random
import pickle
import torch
import numpy as np

from .common import BaseDataset

class Inter4kTrain(BaseDataset):
    def __init__(self, data_opt, **kwargs):
        super(Inter4kTrain, self).__init__(data_opt, **kwargs)

        meta = pickle.load(open(osp.join(self.train_data_dir, 'meta_info.pkl'), 'rb'))
        self.keys = sorted(meta['keys'])

        self.env = self.init_lmdb(self.train_data_dir)
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        key = self.keys[index]
        idx, (n, h, w), frms = self.parse_lmdb_key(key)
        c = 3 if self.train_data_type.lower() == 'rgb' else 1
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

        return pats
