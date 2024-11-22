import os.path as osp
import pickle
import random
import lmdb
import numpy as np
import torch

from .common import BaseDataset
from data.utils import np2Tensor

class Div2kTrain(BaseDataset):
    def __init__(self, data_opt, **kwargs):
        super(Div2kTrain, self).__init__(data_opt, **kwargs)

        meta = pickle.load(
            open(osp.join(self.deg_data_dir, 'meta_info.pkl'), 'rb'))
        self.keys = sorted(meta['keys'])

        self.repeat = 1000 // (len(self.keys) // self.deg_batch_size)

        self.env = self.init_lmdb(self.deg_data_dir)
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, item):
        # if not hasattr(self, 'txn'):
        #     self.open_lmdb()
        
        item = item % len(self.keys)
        key = self.keys[item]
        (h, w), cur_idx = self.parse_image_lmdb_key(key)
        c = 3 if self.deg_data_type.lower() else 1

        img = self.read_lmdb_frame(self.env, key, (h, w, c))

        imgs = self.get_patch(img)
        tsr = [np2Tensor(img) for img in imgs]

        return {'gt': tsr}
    
    # def open_lmdb(self):
    #     self.env = lmdb.open(self.deg_data_dir, readonly=True, lock=True, readahead=False, meminit=False)
    #     self.txn = self.env.begin(buffers=True)
    
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
