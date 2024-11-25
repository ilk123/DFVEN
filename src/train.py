import yaml
import os.path as osp
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning.callbacks as plc

from model import DMInterface, MInterface
from data import DDInterface, DInterface

import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:5930"

def train_degrade(opt):
    pl.seed_everything(opt['seed'])

    d_callbacks = []
    d_callbacks.append(plc.ModelCheckpoint(
        dirpath=opt['logger_ckpt_dir'], 
        filename='d_{epoch:03d}_con_reg', 
        save_top_k=1, 
        every_n_epochs=20, 
        save_on_train_epoch_end=True, 
        enable_version_counter=True
    ))
    d_callbacks.append(plc.LearningRateMonitor(
        logging_interval='epoch'
    ))

    if opt['logger'] == 'tensorboard':
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=opt['logger_dir'], 
            name=None, 
            log_graph=True, 
            version=opt['degrade_type']
        )
    
    d_data_module = DDInterface(**opt)
    d_model = DMInterface(**opt)

    d_trainer = Trainer(accelerator=opt['accelerator'], devices=opt['device'], logger=tb_logger, callbacks=d_callbacks, fast_dev_run=False, max_epochs=opt['degrade_epoch'], log_every_n_steps=opt['deg_logger_freq'])

    d_trainer.fit(d_model, d_data_module)

def train(opt):
    callbacks = []
    callbacks.append(plc.ModelCheckpoint(
        dirpath=opt['logger_ckpt_dir'],
        filename='{epoch:02d}', 
        save_top_k=1, 
        every_n_epochs=1, 
        save_on_train_epoch_end=True, 
        enable_version_counter=True
    ))
    callbacks.append(plc.LearningRateMonitor(
        logging_interval='epoch'
    ))
    
    if opt['logger'] == 'tensorboard':
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=opt['logger_dir'], 
            name=None, 
            version=opt['degrade_type'], 
            log_graph=False
        )

    data_module1 = DInterface(**opt, **{'train_data_name': opt['train_data_name1'], 'train_data_dir': opt['train_data_dir1'], 'train_batch_size': opt['train_batch_size1']})
    model1 = MInterface(**opt)

    trainer1 = Trainer(accelerator=opt['accelerator'], devices=opt['device'], logger=tb_logger, callbacks=callbacks, max_epochs=opt['generator_epoch1'], log_every_n_steps=opt['gen_logger_freq'], val_check_interval=opt['interval'], num_sanity_val_steps=0)
    
    trainer1.fit(model1, data_module1)
    trainer1.save_checkpoint(osp.join(opt['temp_ckpt_dir'], 'temp.ckpt'))

    data_module2 = DInterface(**opt, **{'train_data_name': opt['train_data_name2'], 'train_data_dir': opt['train_data_dir2'], 'train_batch_size': opt['train_batch_size2']})
    model2 = MInterface(**opt)

    trainer2 = Trainer(accelerator=opt['accelerator'], devices=opt['device'], logger=tb_logger, callbacks=callbacks, max_epochs=opt['generator_epoch1'] + opt['generator_epoch2'], log_every_n_steps=opt['gen_logger_freq'], val_check_interval=opt['interval'], num_sanity_val_steps=0)

    if opt['temp_ckpt_dir'] is not None and os.path.exists(opt['temp_ckpt_dir']):
        trainer2.fit(model2, data_module2, ckpt_path=opt['temp_ckpt_dir'])

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--device', default=0, type=int, help='GPU number (-1 for CPU only)')
    parser.add_argument('--opt_yaml', default='single_degradation.yaml', type=str)
    parser.add_argument('--mode', default='train_degrade', type=str, help='train or train_degrade')
    parser.add_argument('--degrade_type', default='noise', type=str, help='degradation type (blur, noise, red, light, and all)')
    parser.add_argument('--model', default='DNet', type=str, help='model to run')

    args = parser.parse_args()

    with open(osp.join('./yaml', args.opt_yaml), 'r') as f:
        opt = yaml.load(f.read(), Loader=yaml.FullLoader)

    ckpt_dir = opt['logger_ckpt_dir']
    if not osp.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if args.mode == 'train':
        opt['gen_model_name'] = args.model
    elif args.mode == 'train_degrade':
        opt['deg_model_name'] = args.model
    
    if args.device >= 0:
        opt['accelerator'] = 'gpu'
    else:
        opt['accelerator'] = 'cpu'

    opt['device'] = [args.device]
    opt['degrade_type'] = args.degrade_type
    opt['batch_loss_file'] = f'./log/single_degradation/{args.degrade_type}/batch_loss.txt'
    opt['epoch_loss_file'] = f'./log/single_degradation/{args.degrade_type}/epoch_loss.txt'

    if args.mode == 'train':
        opt['is_train'] = True
        train(opt)
    elif args.mode == 'train_degrade':
        opt['is_train'] = True
        train_degrade(opt)
    else:
        raise ValueError(
            'Unrecognized mode: {} (train|train_degrade)'.format(args.mode))
