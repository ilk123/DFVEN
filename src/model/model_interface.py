import os.path as osp
import inspect
import importlib
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl

from .loss import CharbonnierLoss
from .metrics import *
from data.degrade.degradation import *
from .utils import patch_clip
from data.utils import pad_sequence, float32_to_uint8
from utils import save_seqence, save_loss


class MInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

        self.example_input_array = self.model.generate_dummy_input((self.hparams.tempo_range, 3, self.hparams.patch_size, self.hparams.patch_size), is_seq=True)
        self.metrics = OrderedDict({metric: [] for metric in self.hparams.metrics})

        self.epoch_count = 0

    def forward(self, lr):
        return self.model.infer(lr)
    
    def on_train_start(self):
        if self.hparams.degrade_type == 'all' and self.hparams.deg_pth_dir is not None:
            if self.global_step == 0:
                D_state_dict1 = torch.load(self.hparams.deg_pth_dir[0])
                GD_state_dict1 = {k: v for k, v in D_state_dict1.items() if k in self.model.Dnet1.state_dict()}
                self.model.Dnet1.load_state_dict(GD_state_dict1)

                D_state_dict2 = torch.load(self.hparams.deg_pth_dir[1])
                GD_state_dict2 = {k: v for k, v in D_state_dict2.items() if k in self.model.Dnet2.state_dict()}
                self.model.Dnet2.load_state_dict(GD_state_dict2)

                D_state_dict3 = torch.load(self.hparams.deg_pth_dir[2])
                GD_state_dict3 = {k: v for k, v in D_state_dict3.items() if k in self.model.Dnet3.state_dict()}
                self.model.Dnet3.load_state_dict(GD_state_dict3)

                D_state_dict4 = torch.load(self.hparams.deg_pth_dir[3])
                GD_state_dict4 = {k: v for k, v in D_state_dict4.items() if k in self.model.Dnet4.state_dict()}
                self.model.Dnet4.load_state_dict(GD_state_dict4)
            
            self.model.Dnet1.queue_ptr = torch.zeros(1, dtype=torch.long)
            self.model.Dnet2.queue_ptr = torch.zeros(1, dtype=torch.long)
            self.model.Dnet3.queue_ptr = torch.zeros(1, dtype=torch.long)
            self.model.Dnet4.queue_ptr = torch.zeros(1, dtype=torch.long)
        
        elif self.hparams.deg_pth_dir is not None:
            if self.global_step == 0:
                D_state_dict = torch.load(self.hparams.deg_pth_dir[0])
                GD_state_dict = {k: v for k, v in D_state_dict.items() if k in self.model.Dnet.state_dict()}
                self.model.Dnet.load_state_dict(GD_state_dict)
            
            self.model.Dnet.queue_ptr = torch.zeros(1, dtype=torch.long)
    
    def on_train_epoch_start(self):
        self.degrade_data_generator = DataPrepare(self.device, self.hparams, self.hparams.degrade_type)
    
    def training_step(self, batch, batch_idx):
        batch = self.degrade_data_generator(batch)
        lr, gt, param = batch['lr'], batch['gt'], batch['para']
        n, t, c, H, W = gt.size()
        p, s = self.hparams.patch_size, self.hparams.scale

        output_dict = self.model.forward(lr)
        hr = output_dict['hr']
        patch_pos = output_dict['patch_pos']

        gt_patch = patch_clip(gt, patch_pos, s, p, is_hr=True)[:, -1, ...] # ntcRR
        lr_patch = patch_clip(lr, patch_pos, s, p, is_hr=False)[:, -1, ...] # ntcrr

        Loss = 0.0
        self.Losses = {}
        for k, v in self.loss_function.items():
            if k == 'pixel':
                p_loss = v[0] * v[1](hr, gt_patch)
                Loss += p_loss
                self.Losses['p_loss'] = p_loss
            elif k == 'degrade':
                output = output_dict['logits']
                target = output_dict['labels']
                d_loss = 0.0
                for i in range(self.hparams.deg_num):
                    d_loss += v[0] * v[1](output[i], target[i]) / self.hparams.deg_num
                Loss += d_loss
                self.Losses['d_loss'] = d_loss
            else:
                raise ValueError("Invalid Loss Type!")
        self.Losses['Loss'] = Loss
        
        self.log_dict(self.Losses, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.logger.experiment.add_image('image/hr0', hr[0, ...], batch_idx)
        self.logger.experiment.add_image('image/lr0', lr_patch[0, ...], batch_idx)
        self.logger.experiment.add_image('image/gt0', gt_patch[0, ...], batch_idx)

        return Loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        save_loss(self.hparams.batch_loss_file, self.global_step // 2, self.Losses)
    
    def on_train_epoch_end(self):
        if self.epoch_count % 5 == 0:
            torch.save(self.model.state_dict(), self.hparams.logger_ckpt_dir + '/DAVSR_{:02d}.pth'.format(self.epoch_count))
        
        save_loss(self.hparams.epoch_loss_file, self.epoch_count, self.Losses)
        
        self.epoch_count += 1
        
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        lr, GT = batch['lr'], batch['gt']
        lr = lr[0].permute(0, 3, 1, 2)
        t, _, _, _ = lr.shape
        lr, n_pad_front = pad_sequence(lr, self.hparams.padding_mode, self.hparams.n_pad_front)

        gt_pre, hr_pre = None, None
        for i in range(t):
            lr_clip = lr[i: i + 4]
            hr = self.model.infer(lr_clip.unsqueeze(dim=0))

            gt = GT.squeeze().cpu().numpy()
            hr = hr.cpu().numpy()
            hr = float32_to_uint8(hr).transpose(0, 2, 3, 1)

            # crop the larger one to match the smaller one
            t, H, W, _ = hr.shape
            _, _, GH, GW = gt.shape
            H, W = min(H, GH), min(W, GW)
            hr, gt = hr[:, :, :H, :W], gt[:, :, :H, :W]

            gt_cur, hr_cur = gt[i, ...], hr[0]
            self.evaluation(gt_cur, hr_cur, gt_pre, hr_pre)

            gt_pre = gt_cur
            hr_pre = hr_cur
        
        self.avg_metrics = {}
        for metric_type, metric in self.metrics.items():
            avg_metric = np.mean(metric)
            self.avg_metrics[metric_type] = avg_metric
        
        self.log_dict(self.avg_metrics, batch_size=1, on_epoch=True, prog_bar=True, logger=True)
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        lr, GT = batch['lr'], batch['gt']
        lr = lr[0].permute(0, 3, 1, 2)
        t, _, _, _ = lr.shape
        lr, n_pad_front = pad_sequence(lr, self.hparams.padding_mode, self.hparams.n_pad_front)

        HR = []
        gt_pre, hr_pre = None, None
        for i in range(t):
            lr_clip = lr[i: i + 4]
            hr = self.model.infer(lr_clip.unsqueeze(dim=0))

            gt = GT.squeeze().cpu().numpy()
            hr = hr.cpu().numpy()
            hr = float32_to_uint8(hr).transpose(0, 2, 3, 1)

            # crop the larger one to match the smaller one
            t, H, W, _ = hr.shape
            _, _, GH, GW = gt.shape
            H, W = min(H, GH), min(W, GW)
            hr, gt = hr[:, :, :H, :W], gt[:, :, :H, :W]

            HR.append(hr[0])
            gt_cur, hr_cur = gt[i, ...], hr[0]
            self.evaluation(gt_cur, hr_cur, gt_pre, hr_pre)

            gt_pre = gt_cur
            hr_pre = hr_cur
        
        return {'loss': 0.0, 'gt': gt, 'hr': np.array(HR)}
    
    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.metric = {}
        self.avg_metrics = {}
        for metric_type, metric in self.metrics.items():
            avg_metric = np.mean(metric)
            self.metric[metric_type] = metric
            self.avg_metrics[metric_type] = avg_metric

        with open(self.hparams.metrics_file, 'a') as f:
            for k, v in self.metrics.items():
                f.write(k + ' ' + str(v) + '\n')
            for k, v in self.avg_metrics.items():
                f.write(k + ' ' + str(v) + '\n')
            f.write('\n')
        
        hr, gt = outputs['hr'], outputs['gt']
        res = []
        for i in range(len(hr)):
            res.append(cv2.hconcat([hr[i], gt[i]]))
        res = np.array(res)
        save_seqence(osp.join(self.hparams.seq_conbine_file_path, str(batch_idx)), res, to_bgr=True)
        save_seqence(osp.join(self.hparams.seq_file_path, str(batch_idx)), hr, to_bgr=True)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        lr, gt = batch['lr'], batch['gt']
        lr, n_pad_front = pad_sequence(lr[0].permute(0, 3, 1, 2), self.hparams.padding_mode, self.hparams.n_pad_front)
        
        hr = self(lr, is_seq=True, is_train=False)
        hr = hr[n_pad_front:, ...]

        return hr

    def configure_optimizers(self):
        if hasattr(self.hparams, 'gen_weight_decay'):
            weight_decay = self.hparams.gen_weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.gen_lr, weight_decay=weight_decay)

        if self.hparams.gen_scheduler is None or self.hparams.gen_scheduler == 'FixedLR':
            return optimizer
        else:
            if self.hparams.gen_scheduler == 'StepLR':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.gen_step_size,
                                       gamma=self.hparams.gen_gamma)
            elif self.hparams.gen_scheduler == 'MultiStepLR':
                scheduler = lrs.MultiStepLR(optimizer, 
                                            milestones=self.hparams.gen_milestones, 
                                            gamma=self.hparams.gen_gamma)
            elif self.hparams.gen_scheduler == 'CosineLR':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.gen_step_size,
                                                  eta_min=self.hparams.gen_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            scheduler = {'scheduler': scheduler, 
                         'interval': 'step'}
            return [optimizer], [scheduler]

    def configure_loss(self):
        self.loss_function = {}
        Loss = self.hparams.gen_loss
        losses = Loss.split('+')
        for w_loss in losses:
            w, loss = w_loss.split('*')
            if loss == 'pixel':
                self.loss_function['pixel'] = [float(w), CharbonnierLoss]
            elif loss == 'degrade':
                self.loss_function['degrade'] = [float(w), F.cross_entropy]
            else:
                raise ValueError("Invalid Loss Type!")
    
    def evaluation(self, gt_cur, hr_cur, gt_pre, hr_pre):
        metrics = self.hparams.metrics
        for metric in metrics:
            if metric.lower() == 'psnr':
                self.metrics[metric].append(compute_PSNR(gt_cur, hr_cur, self.hparams.metrics_colorspace))
            elif metric.lower() == 'ssim':
                self.metrics[metric].append(compute_SSIM(gt_cur, hr_cur, self.hparams.metrics_colorspace))
            elif metric.lower() == 'lpips':
                self.metrics[metric].append(compute_LPIPS(gt_cur, hr_cur)[0, 0, 0, 0])
            elif metric.lower() == 'tof':
                if gt_pre is not None:
                    self.metrics[metric].append(compute_tOF(gt_cur, hr_cur, gt_pre, hr_pre))
            elif metric.lower() == 'tlp':
                if gt_pre is not None:
                    self.metrics[metric].append(compute_tLP(gt_cur, hr_cur, gt_pre, hr_pre))
            else:
                raise ValueError("Invalid Metric Type!")

    def load_model(self):
        name = self.hparams.gen_model_name
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        self.hparams.update(other_args)
        return Model(self.hparams)
