import os.path as osp
import inspect
import importlib
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl

from .loss import CharbonnierLoss
from .metrics import *
from data.degrade.degradation import *
from .utils import patch_clip, backward_warp
from data.utils import pad_sequence, float32_to_uint8
from utils import save_image, show_degrade, count_param, save_loss

class DMInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

        self.example_input_array = {
            'im_q': torch.rand(self.hparams.deg_batch_size, 3, self.hparams.patch_size, self.hparams.patch_size, dtype=torch.float32).cuda(), 
            'im_k': torch.rand(self.hparams.deg_batch_size, 3, self.hparams.patch_size, self.hparams.patch_size, dtype=torch.float32).cuda(), 
            'p': torch.Tensor([1.0]).cuda(), 
        }
        self.degrade_type = ['noise', 'blur', 'light', 'red']

    def forward(self, im_q, im_k, p, is_train=False):
        return self.model(im_q, im_k, p, is_train=False)
    
    def on_train_start(self):
        # self.logger.experiment.add_graph(self.model, self.example_input_array.values())
        self.logger.experiment.add_graph(self.model.encoder_q, self.example_input_array['im_q'])

        count_param(self.model)
    
    def on_train_epoch_start(self):
        self.data_type = self.degrade_type[self.hparams.i]
        self.degrade_data_generator = DataPrepare(self.device, self.hparams, self.data_type)
    
    def on_train_batch_start(self, batch, batch_idx):
        pass
    
    def training_step(self, batch, batch_idx):
        data = [data for data in batch['gt']]
        data = torch.stack(data).permute(1, 0, 2, 3, 4)
        data = {'gt': data}
        batch = self.degrade_data_generator(data)

        data, param = batch['lr'], batch['para']
        embedding, degrade, output, target = self.model.forward(data[:, 0, ...], data[:, 1, ...], param, is_train=True)

        Loss = 0.0
        self.Losses = {}
        for k, v in self.loss_function.items():
            if k == 'contrastive':
                c_loss = v[0] * v[1](output, target)
                Loss += c_loss
                self.Losses['c_loss'] = c_loss
            elif k == 'regress':
                r_loss = v[0] * v[1](param, degrade)
                Loss += r_loss
                self.Losses['r_loss'] = r_loss
            else:
                raise ValueError("Invalid Loss Type!")
        self.Losses['loss'] = Loss
        self.log_dict(self.Losses, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # Loss = self.loss_function(output, target)
        # self.log('d_loss', Loss, on_step=True, on_epoch=True, prog_bar=True)

        em_data = data[:, 0, ...].mul(1/255)
        self.embeddings = {'feature': embedding, 'meta_data': param.cpu().numpy().tolist(), 'label_img': em_data}
        return {'loss': Loss.float()}
    
    def on_train_epoch_end(self):
        pass
        # self.logger.experiment.add_embedding(self.embeddings['feature'], self.embeddings['meta_data'], self.embeddings['label_img'], self.current_epoch, 'embedding')
    
    def on_train_end(self):
        torch.save(self.model.state_dict(), self.hparams.gen_lp[self.hparams.i])
    
    def on_test_start(self):
        embedding = []
        param = []
        self.embeddings = {'embedding': embedding, 'param': param}

    def on_test_epoch_start(self):
        self.data_type = self.degrade_type[self.hparams.i]
        self.degrade_data_generator = DataPrepareTest(self.device, self.hparams, self.data_type)
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        data = [data for data in batch['gt']]
        data = {'gt': torch.stack(data).unsqueeze(0)}
        batch = self.degrade_data_generator(data)

        data, param = batch['lr'], batch['para']
        embedding = self.model.forward(data[0, ...], data[0, ...], param, is_train=False).squeeze()
        # em_data = data[:, 0, ...].mul(1/255)
        # self.embeddings = {'feature': embedding, 'meta_data': param.tolist(), 'label_img': em_data}
        self.embeddings['param'].append(param.cpu().numpy())
        self.embeddings['embedding'].append(embedding.cpu().numpy())
        # self.log('embedding', embedding, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('embedding', param, on_step=True, on_epoch=True, prog_bar=True)
    
    def on_test_end(self):
        show_degrade(self.embeddings)

    def load_model(self):
        name = self.hparams.deg_model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model, deg_dim=self.hparams.deg_dim // self.hparams.deg_num)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        # inkeys = self.hparams.keys()
        # args1 = {}
        # for arg in class_args:
        #     if arg in inkeys:
        #         args1[arg] = getattr(self.hparams, arg)
        self.hparams.update(other_args)
        return Model(self.hparams)

    def configure_optimizers(self):
        if hasattr(self.hparams, 'deg_weight_decay'):
            weight_decay = self.hparams.deg_weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.deg_lr, weight_decay=weight_decay)

        if self.hparams.deg_scheduler is None or self.hparams.deg_scheduler == 'FixedLR':
            return optimizer
        else:
            if self.hparams.deg_scheduler == 'StepLR':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.deg_step_size,
                                       gamma=self.hparams.deg_gamma)
            elif self.hparams.deg_scheduler == 'MultiStepLR':
                scheduler = lrs.MultiStepLR(optimizer, 
                                            milestones=self.hparams.deg_milestones, 
                                            gamma=self.hparams.deg_gamma)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            scheduler = {'scheduler': scheduler, 
                         'interval': 'step'}
            return [optimizer], [scheduler]

    def configure_loss(self):        
        self.loss_function = {}
        loss = self.hparams.deg_loss.lower()
        losses = loss.split('+')
        for w_loss in losses:
            w, loss = w_loss.split('*')
            if loss == 'contrastive':
                self.loss_function['contrastive'] = [float(w), F.cross_entropy]
                # self.loss_function['contrastive'] = [float(w), CosineSimilarityLoss]
                # self.loss_function['contrastive'] = [float(w), F.binary_cross_entropy_with_logits]
            elif loss == 'regress':
                self.loss_function['regress'] = [float(w), F.mse_loss]
            else:
                raise ValueError("Invalid Loss Type!")


class MInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

        self.example_input_array = self.model.generate_dummy_input((3, self.hparams.patch_size, self.hparams.patch_size))
        self.metrics = OrderedDict({metric: [] for metric in self.hparams.metrics})

        self.epoch_count = 0

    def forward(self, lr, is_train=False):
            return self.model.infer_single(lr, is_train=is_train)
    
    def on_train_start(self):
        if self.global_step == 0:
            D_state_dict1 = torch.load(self.hparams.gen_lp[0])
            GD_state_dict1 = {k: v for k, v in D_state_dict1.items() if k in self.model.Dnet1.state_dict()}
            self.model.Dnet1.load_state_dict(GD_state_dict1)

            D_state_dict2 = torch.load(self.hparams.gen_lp[1])
            GD_state_dict2 = {k: v for k, v in D_state_dict2.items() if k in self.model.Dnet2.state_dict()}
            self.model.Dnet2.load_state_dict(GD_state_dict2)

            D_state_dict3 = torch.load(self.hparams.gen_lp[2])
            GD_state_dict3 = {k: v for k, v in D_state_dict3.items() if k in self.model.Dnet3.state_dict()}
            self.model.Dnet3.load_state_dict(GD_state_dict3)

            D_state_dict4 = torch.load(self.hparams.gen_lp[3])
            GD_state_dict4 = {k: v for k, v in D_state_dict4.items() if k in self.model.Dnet4.state_dict()}
            self.model.Dnet4.load_state_dict(GD_state_dict4)
        
        self.model.Dnet1.queue_ptr = torch.zeros(1, dtype=torch.long)
        self.model.Dnet2.queue_ptr = torch.zeros(1, dtype=torch.long)
        self.model.Dnet3.queue_ptr = torch.zeros(1, dtype=torch.long)
        self.model.Dnet4.queue_ptr = torch.zeros(1, dtype=torch.long)
    
    def on_train_epoch_start(self):
        self.degrade_data_generator = DataPrepare(self.device, self.hparams, 'total')
    
    def training_step(self, batch, batch_idx):
        batch = self.degrade_data_generator(batch)
        lr, gt, param = batch['lr'], batch['gt'], batch['para']
        p, s = self.hparams.patch_size, self.hparams.scale

        output_dict = self.model.forward_single(lr, is_train=True)
        hr = output_dict['hr']
        patch_pos = output_dict['patch_pos']

        gt_data_patch = patch_clip(gt, patch_pos, s, p, is_hr=True) # ncRR
        lr_data_patch = patch_clip(lr, patch_pos, s, p, is_hr=False) # ncrr

        Loss = 0.0
        self.Losses = {}
        for k, v in self.loss_function.items():
            if k == 'pixel':
                p_loss = v[0] * v[1](hr, gt_data_patch)
                Loss += p_loss
                self.Losses['p_loss'] = p_loss
            elif k == 'warping':
                lr_warp = backward_warp(output_dict['lr_prev'], output_dict['lr_flow'])
                w_loss = v[0] * v[1](lr_warp, output_dict['lr_curr'])
                Loss += w_loss
                self.Losses['w_loss'] = w_loss
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
        self.logger.experiment.add_image('image/hr0', hr[0], batch_idx)
        self.logger.experiment.add_image('image/lr0', lr_data_patch[0], batch_idx)
        self.logger.experiment.add_image('image/gt0', gt_data_patch[0], batch_idx)

        return Loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        save_loss(self.hparams.batch_loss_file, self.global_step // 2, self.Losses)
    
    def on_train_epoch_end(self):
        if self.epoch_count % 5 == 0:
            torch.save(self.model.state_dict(), self.hparams.logger_ckpt_dir + '/DAVSR_{:02d}.pth'.format(self.epoch_count))
        
        save_loss(self.hparams.epoch_loss_file, self.epoch_count, self.Losses)
        
        self.epoch_count += 1
        
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        lr, gt = batch['lr'], batch['gt']
        lr = lr[0].permute(0, 3, 1, 2)
        GT = gt[0]
        t, _, _, _ = GT.shape

        gt_pre, hr_pre = None, None
        for i in range(t):
            hr = self.model.infer_single(lr[i].unsqueeze(dim=0), is_train=False)

            gt = GT[i].cpu().numpy()
            hr = hr.squeeze().cpu().numpy()
            hr = float32_to_uint8(hr).transpose(1, 2, 0)

            # crop the larger one to match the smaller one
            H, W, _ = hr.shape
            _, GH, GW = gt.shape
            H, W = min(H, GH), min(W, GW)
            hr, gt = hr[:, :H, :W], gt[:, :H, :W]

            gt_cur, hr_cur = gt, hr
            self.evaluation(gt_cur, hr_cur, gt_pre, hr_pre)

            gt_pre = gt_cur
            hr_pre = hr_cur
        
        self.avg_metrics = {}
        for metric_type, metric in self.metrics.items():
            avg_metric = np.mean(metric)
            self.avg_metrics[metric_type] = avg_metric
        
        self.log_dict(self.avg_metrics, batch_size=1, on_epoch=True, prog_bar=True, logger=True)
    
    # def on_test_epoch_start(self):
        # self.data_type = 'without_red'
        # self.degrade_data_generator = DataPrepareTest(self.device, self.hparams, self.data_type)
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        lr, gt = batch['lr'], batch['gt']
        lr = lr[0].permute(0, 3, 1, 2)
        GT = gt[0]
        t, _, _, _ = GT.shape

        gt_pre, hr_pre = None, None
        for i in range(t):
            hr = self.model.infer_single(lr[i].unsqueeze(dim=0), is_train=False)

            gt = GT[i].cpu().numpy()
            hr = hr.squeeze().cpu().numpy()
            hr = float32_to_uint8(hr).transpose(1, 2, 0)

            # crop the larger one to match the smaller one
            H, W, _ = hr.shape
            _, GH, GW = gt.shape
            H, W = min(H, GH), min(W, GW)
            hr, gt = hr[:, :H, :W], gt[:, :H, :W]

            gt_cur, hr_cur = gt, hr
            self.evaluation(gt_cur, hr_cur, gt_pre, hr_pre)

            gt_pre = gt_cur
            hr_pre = hr_cur
        
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
        
        return {'loss': 0.0, 'gt': gt, 'hr': hr}
    
    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        hr, gt = outputs['hr'], outputs['gt']
        res = cv2.hconcat([hr, gt])
        save_image(osp.join(self.hparams.seq_conbine_file_path, f'{str(batch_idx)}.png'), res, to_bgr=True)
        save_image(osp.join(self.hparams.seq_file_path, f'{str(batch_idx)}.png'), hr, to_bgr=True)
        # res = []
        # for i in range(len(hr)):
        #     res.append(cv2.hconcat([hr[i], gt[i]]))
        # res = np.array(res)
        # save_seqence(osp.join(self.hparams.seq_conbine_file_path, str(batch_idx)), res, to_bgr=True)
        # save_seqence(osp.join(self.hparams.seq_file_path, str(batch_idx)), hr, to_bgr=True)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        lr, gt = batch['lr'], batch['gt']
        lr = lr[0].permute(0, 3, 1, 2)
        
        hr = self(lr, is_seq=True, is_train=False)

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
            elif loss == 'warping':
                self.loss_function['warping'] = [float(w), CharbonnierLoss]
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
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        # inkeys = self.hparams.keys()
        # args1 = {}
        # for arg in class_args:
        #     if arg in inkeys:
        #         args1[arg] = getattr(self.hparams, arg)
        self.hparams.update(other_args)
        return Model(self.hparams)
