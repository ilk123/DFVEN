import yaml
import os.path as osp
from argparse import ArgumentParser

import cv2
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning.callbacks as plc
import tensorboard

from model import DMInterface, MInterface
from data import DDInterface, DInterface
from data.utils import np2Tensor, float32_to_uint8
from model.metrics import compute_SSIM

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
    d_callbacks.append(plc.ModelCheckpoint(
        dirpath=opt['logger_ckpt_dir'], 
        filename='d_best_con_reg', 
        monitor='loss', 
        mode='min'
    ))
    d_callbacks.append(plc.LearningRateMonitor(
        logging_interval='epoch'
    ))

    # for i in range(opt['deg_num']):
    if opt['logger'] == 'tensorboard':
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=opt['logger_dir'], 
            # version=f'{i:03d}', 
            log_graph=True
        )
    
    d_data_module = DDInterface(**opt)
    # d_model = DMInterface(**opt, **{'i': i})
    d_model = DMInterface(**opt)

    d_trainer = Trainer(accelerator=opt['accelerator'], devices=1, logger=tb_logger, callbacks=d_callbacks, fast_dev_run=False, max_epochs=opt['degrade_epoch'] // opt['deg_num'], log_every_n_steps=opt['deg_logger_freq'])

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
    # callbacks.append(plc.EarlyStopping(
    #     monitor='SSIM', 
    #     mode='max',
    #     patience=2,
    #     stopping_threshold=0.97, 
    #     check_on_train_epoch_end=True
    # ))
    
    if opt['logger'] == 'tensorboard':
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=opt['logger_dir'], 
            # version=f'{(i+1):03d}', 
            version=f'{100}', 
            log_graph=False
        )

    data_module1 = DInterface(**opt, **{'train_data_name': opt['train_data_name1'], 'train_data_dir': opt['train_data_dir1'], 'train_batch_size': opt['train_batch_size1']})
    model1 = MInterface(**opt)

    trainer1 = Trainer(accelerator=opt['accelerator'], devices=[0], logger=tb_logger, callbacks=callbacks, max_epochs=opt['generator_epoch1'], log_every_n_steps=opt['gen_logger_freq'], val_check_interval=opt['interval'], num_sanity_val_steps=0)
    
    trainer1.fit(model1, data_module1)
    trainer1.save_checkpoint(osp.join(opt['logger_ckpt_dir'], '6.ckpt'))

    opt['deg_batch_size'] = opt['deg_batch_size'] * 2
    data_module2 = DInterface(**opt, **{'train_data_name': opt['train_data_name2'], 'train_data_dir': opt['train_data_dir2'], 'train_batch_size': opt['train_batch_size2']})
    model2 = MInterface(**opt)

    trainer2 = Trainer(accelerator=opt['accelerator'], devices=[0], logger=tb_logger, callbacks=callbacks, max_epochs=opt['generator_epoch1'] + opt['generator_epoch2'], log_every_n_steps=opt['gen_logger_freq'], val_check_interval=opt['interval'], num_sanity_val_steps=0)

    if opt['gen_load_path'] is not None and os.path.exists(opt['gen_load_path']):
        trainer2.fit(model2, data_module2, ckpt_path=opt['gen_load_path'])

#测试评价指标，使用xxx_test.yaml 
def test(opt):
    if opt['logger'] == 'tensorboard':
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=opt['logger_dir'], 
            version='test', 
            log_graph=False
        )

    data_module = DInterface(**opt, **{'train_data_name': opt['train_data_name2'], 'train_data_dir': opt['train_data_dir2']})
    model = MInterface(**opt)
    trainer = Trainer(accelerator=opt['accelerator'], devices=[0], logger=tb_logger)
    trainer.test(model, data_module, opt['gen_load_path'])

def test_degrade(opt):
    if opt['logger'] == 'tensorboard':
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=opt['logger_dir'], 
            version='test_degrade', 
            log_graph=True
        )
    data_module = DDInterface(**opt)
    # model = DMInterface(**opt, **{'i': 0})
    model = DMInterface(**opt)
    trainer = Trainer(accelerator=opt['accelerator'], devices=1, logger=tb_logger)
    # trainer.test(model, data_module)
    trainer.test(model, data_module, opt['deg_load_path'])

# 测试视频并保存视频
def predict(opt):
    model = MInterface.load_from_checkpoint(opt['gen_load_path'])
    model.eval()
    device = torch.device('cuda:0')

    for i in range(1, 7):
        video_path = f'/home/xhd/PELD/enhancement/data/video/Encode_1080P_{i}.mp4'
        output_path = f'./output/Encode_1080P_{i}.mp4'
        img_size = (3840, 2160)

        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(output_path, fourcc, 30, img_size)

        data = {}
        j = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                input = np2Tensor(frame).unsqueeze(0).to(device)
                if j == 0:
                    data['lr_prev'] = input
                    data['hr_prev'] = np2Tensor(cv2.resize(frame, dsize=None, fx=2, fy=2)).unsqueeze(0).to(device)
                data['lr_curr'] = input
                
                with torch.no_grad():
                    output_tensor = model(data, is_seq=False)
                output = float32_to_uint8(output_tensor.cpu().numpy()).transpose(0, 2, 3, 1).squeeze(0)

                out.write(output)
                cv2.imshow('1', output)
                key = cv2.waitKey(1)
                if key == 32:
                    break

                j += 1
                data['lr_prev'] = input
                data['hr_prev'] = output_tensor
                print(i, j)
            
            else:
                cap.release()
                out.release()
        out.release()
        cap.release()
    
    out.release()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--opt_yaml', default='total-4x-no-seg.yaml', type=str)
    parser.add_argument('--mode', default='train', type=str)
    args = parser.parse_args()

    with open(osp.join('./yaml', args.opt_yaml), 'r') as f:
        opt = yaml.load(f.read(), Loader=yaml.FullLoader)

    ckpt_dir = opt['logger_ckpt_dir']
    if not osp.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    if opt['gpu'] >= 0:
        opt['accelerator'] = 'gpu'
    else:
        opt['accelerator'] = 'cpu'

    if args.mode == 'train':
        opt['is_train'] = True
        train(opt)
    elif args.mode == 'test':
        opt['is_train'] = False
        test(opt)
    elif args.mode == 'predict':
        opt['is_train'] = False
        predict(opt)
    elif args.mode == 'train_degrade':
        opt['is_train'] = True
        train_degrade(opt)
    elif args.mode == 'test_degrade':
        opt['is_train'] = False
        test_degrade(opt)
    else:
        raise ValueError(
            'Unrecognized mode: {} (train|test)'.format(args.mode))
