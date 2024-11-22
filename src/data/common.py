import lmdb
import numpy as np
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, data_opt, **kwargs):
        for kw, args in data_opt.items():
            setattr(self, kw, args)
        
        for kw, args in kwargs.items():
            setattr(self, kw, args)
    
    def len(self):
        pass

    def __getitem__(self):
        pass

    def check_info(self, gt_keys, lr_keys):
        if len(gt_keys) != len(lr_keys):
            raise ValueError(
                'GT & LR contain different numbers of images ({}  vs. {})'.format(
                    len(gt_keys), len(lr_keys)))
        
        for i, (gt_key, lr_key) in enumerate(zip(gt_keys, lr_keys)):
            gt_info = self.parse_lmdb_key(gt_key)
            lr_info = self.parse_lmdb_key(lr_key)

            if gt_info[0] !=lr_info[0]:
                raise ValueError(
                    'video index mismatch ({} vs. {} for the {} key)'.format(
                        gt_info[0], lr_info[0], i))
            
            gt_num, gt_h, gt_w = gt_info[1]
            lr_num, lr_h, lr_w = lr_info[1]
            s = self.scale
            if (gt_num != lr_num) or (gt_h != lr_h * s) or (gt_w != lr_w * s):
                raise ValueError(
                    'video size mismatch ({} vs. {} for the {} key)'.format(
                        gt_info[1], lr_info[1], i))
            if gt_info[2] != lr_info[2]:
                raise ValueError(
                    'frame mismatch ({} vs. {} for the {} key)'.format(
                        gt_info[2], lr_info[2], i))
    
    @staticmethod
    def init_lmdb(seq_dir):
        env = lmdb.open(
            seq_dir, readonly=True, lock=True, readahead=False, meminit=False)
        return env

    @staticmethod
    def parse_lmdb_key(key):
        key_lst = key.split('_')
        idx, size, frms = key_lst[:-2], key_lst[-2], key_lst[-1]
        idx = '_'.join(idx)
        size = tuple(map(int, size.split('x'))) # n, h, w
        frm_start = int(frms.split('-')[0])
        frm_end = int(frms.split('-')[1]) + 1
        frms = tuple(i for i in range(frm_start, frm_end))
        # idx, size, frm = key_lst[:-2], key_lst[-2], int(key_lst[-1])
        # idx = '_'.join(idx)
        # size = tuple(map(int, size.split('x'))) # n_frm, h, w
        return idx, size, frms
    
    @staticmethod
    def parse_image_lmdb_key(key):
        key_lst = key.split('_')
        size, idx = key_lst[0], int(key_lst[-1])
        size = tuple(map(int, size.split('x'))) # h, w
        return size, idx
    
    @staticmethod
    def read_lmdb_frame(env, key, size):
        with env.begin(write=False) as txn:
            buf = txn.get(key.encode('ascii'))
        frms = np.frombuffer(buf, dtype=np.uint8).reshape(*size)
        return frms
    
    @staticmethod
    def augment_sequence(**kwargs):
        pass
    
    def crop_sequence(self, **kwargs):
        pass
