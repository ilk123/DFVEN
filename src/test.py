import yaml
import os.path as osp
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers

from model import DMInterface, MInterface
from data import DDInterface, DInterface

import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:5930"

def test(opt):
    if opt['logger'] == 'tensorboard':
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=opt['logger_dir'], 
            name=None, 
            version=opt['degrade_type']+'/test', 
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
            name=None, 
            version=opt['degrade_type']+'/test_degrade', 
            log_graph=True
        )
    data_module = DDInterface(**opt)
    model = DMInterface(**opt)
    trainer = Trainer(accelerator=opt['accelerator'], devices=1, logger=tb_logger)
    trainer.test(model, data_module, opt['deg_load_path'])

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--device', default=0, type=int, help='GPU number (-1 for CPU only)')
    parser.add_argument('--opt_yaml', default='single_degradation.yaml', type=str)
    parser.add_argument('--mode', default='test', type=str, help='test or test_degrade')
    parser.add_argument('--degrade_type', default='all', type=str, help='degradation type (blur, noise, red, light, and all)')
    parser.add_argument('--model', default='DFVEN', type=str, help='model to run')
    parser.add_argument('--pertrained_model', default='', type=str, help='pertrained model')
    parser.add_argument('--test_lr_path', default='', type=str)
    parser.add_argument('--test_gt_path', default='', type=str)

    args = parser.parse_args()

    with open(osp.join('./yaml', args.opt_yaml), 'r') as f:
        opt = yaml.load(f.read(), Loader=yaml.FullLoader)

    ckpt_dir = opt['logger_ckpt_dir']
    if not osp.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if args.mode == 'test':
        opt['gen_model_name'] = args.model
        opt['gen_load_path'] = args.pertrained_model
    elif args.mode == 'test_degrade':
        opt['deg_model_name'] = args.model
        opt['deg_load_path'] = args.pertrained_model
    
    if opt['gpu'] >= 0:
        opt['accelerator'] = 'gpu'
    else:
        opt['accelerator'] = 'cpu'

    opt['test_lr_dir'] = args.test_lr_path
    opt['test_gt_dir'] = args.test_gt_path
    opt['metrics_file'] = f'./log/single_degradation/{args.degrade_type}/test/metrics.txt'
    opt['seq_file_path'] = f'./log/single_degradation/{args.degrade_type}/test/results'
    opt['seq_conbine_file_path'] = f'./log/single_degradation/{args.degrade_type}/test/results/combine'

    if args.mode == 'test':
        opt['is_train'] = False
        test(opt)
    elif args.mode == 'test_degrade':
        opt['is_train'] = False
        test_degrade(opt)
    else:
        raise ValueError(
            'Unrecognized mode: {} (test|test_degrade)'.format(args.mode))