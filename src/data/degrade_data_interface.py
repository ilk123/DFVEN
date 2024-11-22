import importlib
import pytorch_lightning as pl
from torch.utils.data import DataLoader, _utils


class DDInterface(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.load_data_module()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.trainset = self.instancialize(train=True)
        elif stage == 'test':
            self.testset = self.instancialize(train=False)
    
    def train_dataloader(self):
        return DataLoader(
            self.trainset, 
            batch_size=self.hparams.deg_batch_size, 
            shuffle=True, 
            num_workers=self.hparams.deg_num_workers, 
            pin_memory=self.hparams.deg_pin_memory, 
            drop_last=True, 
            collate_fn=_utils.collate.default_collate, 
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.testset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=self.hparams.deg_num_workers, 
            pin_memory=self.hparams.deg_pin_memory
        )    
    
    def load_data_module(self):
        train_name = '_'.join([self.hparams.deg_data_name, 'train'])
        test_name = '_'.join([self.hparams.deg_data_name, 'test'])
        train_camel_name = ''.join([i.capitalize() for i in train_name.split('_')])
        test_camel_name = ''.join([i.capitalize() for i in test_name.split('_')])
        try:
            self.train_data_module = getattr(importlib.import_module(
                '.'+train_name, package=__package__), train_camel_name)
            self.test_data_module = getattr(importlib.import_module(
                '.'+test_name, package=__package__), test_camel_name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name Data. {train_name}.{train_camel_name} {test_name}.{test_camel_name}')
    
    def instancialize(self, train=True, **other_args):
        self.hparams.update(other_args)
        if train:
            return self.train_data_module(data_opt=self.hparams)
        else:
            return self.test_data_module(data_opt=self.hparams)