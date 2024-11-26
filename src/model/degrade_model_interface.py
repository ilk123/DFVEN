import importlib

import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl

from data.degrade.degradation import *
from utils import show_degrade

class DMInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

        self.example_input_array = {
            'im_q': torch.rand(self.hparams.deg_batch_size, 3, self.hparams.patch_size, self.hparams.patch_size, dtype=torch.float32).cuda(), 
            'im_k': torch.rand(self.hparams.deg_batch_size, 3, self.hparams.patch_size, self.hparams.patch_size, dtype=torch.float32).cuda(), 
        }

    def forward(self, im_q, im_k, is_train=False):
        return self.model(im_q, im_k, is_train=False)
    
    def on_train_start(self):
        self.logger.experiment.add_graph(self.model.encoder_q, self.example_input_array['im_q'])
    
    def on_train_epoch_start(self):
        self.data_type = self.hparams.degrade_type
        self.degrade_data_generator = DataPrepare(self.device, self.hparams, self.data_type)
    
    def on_train_batch_start(self, batch, batch_idx):
        pass
    
    def training_step(self, batch, batch_idx):
        data = [data for data in batch['gt']]
        data = torch.stack(data).permute(1, 0, 2, 3, 4)
        data = {'gt': data}
        batch = self.degrade_data_generator(data)

        data, param = batch['lr'], batch['para']
        embedding, degrade, output, target = self.model.forward(data[:, 0, ...], data[:, 1, ...], is_train=True)

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

        em_data = data[:, 0, ...].mul(1/255)
        self.embeddings = {'feature': embedding, 'meta_data': param.cpu().numpy().tolist(), 'label_img': em_data}
        return {'loss': Loss.float()}
    
    # def on_train_epoch_end(self):
        # self.logger.experiment.add_embedding(self.embeddings['feature'], self.embeddings['meta_data'], self.embeddings['label_img'], self.current_epoch, 'embedding')
    
    def on_train_end(self):
        torch.save(self.model.state_dict(), self.hparams.deg_pth_dir[0])
    
    def on_test_start(self):
        embedding = []
        param = []
        self.embeddings = {'embedding': embedding, 'param': param}

    def on_test_epoch_start(self):
        self.data_type = self.hparams.degrade_type
        self.degrade_data_generator = DataPrepareTest(self.device, self.hparams, self.data_type)
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        data = [data for data in batch['gt']]
        data = {'gt': torch.stack(data).unsqueeze(0)}
        batch = self.degrade_data_generator(data)

        data, param = batch['lr'], batch['para']
        embedding = self.model.forward(data[0, ...], data[0, ...], is_train=False).squeeze()
        self.embeddings['param'].append(param.cpu().numpy())
        self.embeddings['embedding'].append(embedding.cpu().numpy())
    
    def on_test_end(self):
        show_degrade(self.embeddings)

    def load_model(self):
        name = self.hparams.deg_model_name
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
            elif loss == 'regress':
                self.loss_function['regress'] = [float(w), F.mse_loss]
            else:
                raise ValueError("Invalid Loss Type!")