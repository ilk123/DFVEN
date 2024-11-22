import importlib
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DInterface(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.load_data_module()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.trainset = self.instancialize(train=0)
            self.valset = self.instancialize(train=1)
        
        if stage == 'test' or stage == 'predict':
            self.testset = self.instancialize(train=2)
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset, 
            batch_size=self.hparams.train_batch_size, 
            shuffle=True, 
            num_workers=self.hparams.train_num_workers, 
            pin_memory=self.hparams.train_pin_memory, 
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.valset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=self.hparams.train_num_workers, 
            pin_memory=self.hparams.train_pin_memory
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.testset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=self.hparams.train_num_workers, 
            pin_memory=self.hparams.train_pin_memory
        )    
    
    def load_data_module(self):
        train_name = '_'.join([self.hparams.train_data_name, 'train'])
        val_name = '_'.join([self.hparams.train_data_name, 'val'])
        test_name = '_'.join([self.hparams.train_data_name, 'test'])
        train_camel_name = ''.join([i.capitalize() for i in train_name.split('_')])
        val_camel_name = ''.join([i.capitalize() for i in val_name.split('_')])
        test_camel_name = ''.join([i.capitalize() for i in test_name.split('_')])
        try:
            self.train_data_module = getattr(importlib.import_module(
                '.'+train_name, package=__package__), train_camel_name)
            self.val_data_module = getattr(importlib.import_module(
                '.'+val_name, package=__package__), val_camel_name)
            self.test_data_module = getattr(importlib.import_module(
                '.'+test_name, package=__package__), test_camel_name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name Data. {train_name}.{train_camel_name} {val_name}.{val_camel_name} {test_name}.{test_camel_name}')
    
    def instancialize(self, train=True, **other_args):
        self.hparams.update(other_args)
        if train == 0:
            return self.train_data_module(data_opt=self.hparams)
        elif train == 1:
            return self.val_data_module(data_opt=self.hparams)
        elif train == 2:
            return self.test_data_module(data_opt=self.hparams)
