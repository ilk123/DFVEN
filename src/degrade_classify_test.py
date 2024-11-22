import yaml
import torch
import torch.utils.data.dataset

import pytorch_lightning as pl

from .data.degrade.degradation import *

data_path = 'D:/PELD/data/data/DIV2K/DIV2K_valid_HR'
yaml_path = './yaml/red_gan.yaml'

data_type = 'red'
opt = yaml.load(open(yaml_path), Loader=yaml.FullLoader)
device = torch.device('cuda:0')
degrade_data_generator = DataPrepare(device, opt, data_type)

data = 
data = torch.stack(data).permute(1, 0, 2, 3, 4)
batch = degrade_data_generator(data)
class MyDataset(torch.utils.data.Dataset):#需要继承torch.utils.data.Dataset
    def __init__(self):
        #对继承自父类的属性进行初始化(好像没有这句也可以？？)
        super(MyDataset,self).__init__()
        # TODO
        #1、初始化一些参数和函数，方便在__getitem__函数中调用。
        #2、制作__getitem__函数所要用到的图片和对应标签的list。
        #也就是在这个模块里，我们所做的工作就是初始化该类的一些基本参数。
        pass
    def __getitem__(self, index):
        # TODO
        #1、根据list从文件中读取一个数据（例如，使用numpy.fromfile，PIL.Image.open）。
        #2、预处理数据（例如torchvision.Transform）。
        #3、返回数据对（例如图像和标签）。
        #这里需要注意的是，这步所处理的是index所对应的一个样本。
        pass
    def __len__(self):
        return len()