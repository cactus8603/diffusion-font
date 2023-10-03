import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
import random

import torch
# import math
import numpy as np
# import timm
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import torch.multiprocessing as mp
# import torch.utils.data.distributed
# import torch.optim.lr_scheduler as lr_scheduler
import argparse
# from torch.nn.parallel import DistributedDataParallel as DDP
# from ignite.handlers import create_lr_scheduler_with_warmup
# from tensorboardX import SummaryWriter

from labml import lab, tracker, experiment, monit
# from labml.configs import BaseConfigs, option
# from labml_helpers.device import DeviceConfigs
# from labml_nn.diffusion.ddpm import DenoiseDiffusion
# from labml_nn.diffusion.ddpm.unet import UNet

# from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
# from torch.cuda import amp
# from utils.dataset import ImgDataSet
# from utils.utils import read_spilt_data, get_loader, train_one_epoch, evaluate
import sys

# print(sys.path)
from utils import Configs
# import utils.utils

def create_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config_path", default="config.yaml", nargs='?', help="path to config file")
    parser.add_argument("--data_path", default='./Dataset/byFont', type=str, help='')
    parser.add_argument("--font_classes", default='./cfgs/font_classes_173.json', type=str, help='')

    # ddp setting
    parser.add_argument("--use_ddp", default=False, type=bool, help='use ddp or not')
    parser.add_argument("--port", default=8888, type=int, help='ddp port')

    # training setting
    parser.add_argument("--lr", default=0.01, type=float, help='learning rate')
    parser.add_argument("--epoch", default=5, type=int, help='total epoch')
    parser.add_argument("--n_classes", default=173, type=int, help='total classes')
    parser.add_argument("--n_steps", default=100, type=int, help='')
    parser.add_argument("--n_samples", default=16, type=int, help='Number of samples to generate')
    parser.add_argument("--accumulation_step", default=4, type=int, help='')
    parser.add_argument("--seed", default=8603, type=int, help='init random seed')

    # save and load data path
    parser.add_argument("--model_save_path", default='./result', type=str, help='path to save model')
    parser.add_argument("--save_frequency", default=3, type=int, help='save model frequency')
    parser.add_argument("--dict_path", default='', type=str, help='path to json file')

    # warmup
    parser.add_argument("--warmup", default=False, type=bool, help='use warmup or not')
    parser.add_argument("--warmup_start_value", default=0.001, type=float, help='')
    parser.add_argument("--warmup_step", default=5, type=int, help='warmup steps')

    ### parameter setting ###
    # optim and lr scheduler
    parser.add_argument("--momentum", default=0.937, type=float, help='')
    parser.add_argument("--weight_decay", default=0.00005, type=float, help='')
    parser.add_argument("--lrf", default=0.0005, type=float, help='')
    parser.add_argument("--cosanneal_cycle", default=50, type=int, help='')

    parser.add_argument("--batch_size", default=128, type=int, help='')
    parser.add_argument("--num_workers", default=6, type=int, help='')

    # parser.add_argument("", default=, type=, help='')
    
    args = parser.parse_args()
    return args


# set seed
def init(seed):
    seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def cleanup():
    dist.destroy_process_group()

def is_main_worker(gpu):
    return (gpu <= 0)

# mp.spawn will pass the value "gpu" as rank
def train_ddp(rank, world_size, args):

    port = args.port
    dist.init_process_group(
        backend='nccl',
        init_method="tcp://127.0.0.1:" + str(port),
        world_size=world_size,
        rank=rank,
    )

    train(args, ddp_gpu=rank)
    cleanup()

# train function
def train():

    # Create experiment
    experiment.create(name='diffuse', writers={'tensorboard'})

    # Create configurations
    configs = Configs()

    # Set configurations. You can override the defaults by passing the values in the dictionary.
    experiment.configs(configs, {
        'dataset': 'ImgDataset', # 'MNIST',  # 'ImgDataset'
        'image_channels': 3,  # 1,
        'epochs': 200,  # 5,
    })

    # Initialize
    configs.init()

    # Set models for saving and loading
    experiment.add_pytorch_models({'eps_model': configs.eps_model})

    # Start and run the training loop
    with experiment.start():
        configs.run()

if __name__ == '__main__':

    # get args
    # args = create_parser()

    # init random seed
    # init(args.seed)

    # train in ddp or not
    # if args.use_ddp:
    #     n_gpus_per_node = torch.cuda.device_count()
    #     world_size = n_gpus_per_node
    #     mp.spawn(train_ddp, nprocs=n_gpus_per_node, args=(world_size, args))
    # else:
    #     train()

    train()

    