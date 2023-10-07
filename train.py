import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
import random

import torch
import math
import numpy as np
import timm
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torch.optim.lr_scheduler as lr_scheduler
import argparse
from typing import List
from torch.nn.parallel import DistributedDataParallel as DDP
from ignite.handlers import create_lr_scheduler_with_warmup
from labml_nn.diffusion.ddpm import DenoiseDiffusion
# from labml_nn.diffusion.ddpm.unet import UNet
from tensorboardX import SummaryWriter
# from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
from torch.cuda import amp
from func.dataset import ImgDataSet
from model.UNet import UNet
from func.utils import get_loader, train_one_epoch, load_model, evaluate
from model.diff import Diffusion

def create_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config_path", default="config.yaml", nargs='?', help="path to config file")
    parser.add_argument("--data_path", default='/code/Font/byFont', type=str, help='')
    parser.add_argument("--json_file", default='./cfgs/font_classes_173.json', type=str, help='')

    # ddp setting
    parser.add_argument("--use_ddp", default=True, type=bool, help='use ddp or not')
    parser.add_argument("--port", default=8888, type=int, help='ddp port')

    # training setting
    parser.add_argument("--lr", default=0.01, type=float, help='learning rate')
    parser.add_argument("--epoch", default=5, type=int, help='total epoch')
    parser.add_argument("--n_classes", default=173, type=int, help='total classes')
    parser.add_argument("--n_steps", default=1000, type=int, help='')
    parser.add_argument("--n_samples", default=1, type=int, help='Number of samples to generate, only 1 now')
    parser.add_argument("--accumulation_step", default=4, type=int, help='')
    parser.add_argument("--seed", default=8603, type=int, help='init random seed')

    # save and load data path
    parser.add_argument("--model_save_path", default='./result/no_feature', type=str, help='path to save model')
    parser.add_argument("--save_frequency", default=3, type=int, help='save model frequency')
    parser.add_argument("--dict_path", default='', type=str, help='path to json file')

    # images setting
    parser.add_argument("--image_size", default=224, type=int, help='size of input image')
    parser.add_argument("--image_channels", default=3, type=int, help='RGB')
    parser.add_argument("--n_channels", default=16, type=int, help='Number of channels in the initial feature map')
    parser.add_argument("--is_attn", default=[False, False, False, True], type=List[int], help='')
    parser.add_argument("--ch_mults", default=[1, 2, 2, 4], type=List[int], help='')

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

    parser.add_argument("--batch_size", default=64, type=int, help='')
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
def train(args, ddp_gpu=-1):
    cudnn.benchmark = True

    # set gpu of each multiprocessing
    torch.cuda.set_device(ddp_gpu)
    
    # get dataLoader
    train_loader = get_loader(args) 
    print("Get data loader successfully")

    # load model
    device = torch.device('cuda', ddp_gpu)
    model = UNet(
        image_channels=args.image_channels,
        n_channels=args.n_channels,
        ch_mults=args.ch_mults,
        is_attn=args.is_attn
    ).to(device)

    print("load model successful")

    # for name, param in model.named_parameters():
    #     if param.grad is None:
    #         print(name)
    
    # print(model.requires_grad_)
    # return 0

    # check if folder exist and start summarywriter on main worker
    if is_main_worker(ddp_gpu):
        print("Start Training")
        if not os.path.exists(args.model_save_path):
            os.mkdir(args.model_save_path)
        tb_writer = SummaryWriter(args.model_save_path)

    # setting optim
    pg = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(pg, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # setting lr scheduler as cosine annealing
    lf = lambda x: ((1 + math.cos(x * math.pi / args.cosanneal_cycle)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer=opt, lr_lambda=lf)
    
    # setting warm up info
    if args.warmup :
        warmup = create_lr_scheduler_with_warmup(
            scheduler, 
            warmup_start_value=args.warmup_start_value,
            warmup_end_value=args.lr,
            warmup_duration=args.warmup_step,
        )

    start_epoch = 0

    # setting Distributed 
    if args.use_ddp:   
        model = DDP(model.to(ddp_gpu), find_unused_parameters=True)
    else:
        model = model.to(ddp_gpu)
    
    score = 0
    
    # setting Automatic mixed precision
    scaler = amp.GradScaler()

    # start training
    for epoch in range(start_epoch, args.epoch):
        
        diffusion = DenoiseDiffusion(
            eps_model=model,
            n_steps=args.n_steps,
            device=device
        )

        # train 
        train_loss = train_one_epoch(
            model=model, 
            diffusion=diffusion,
            optimizer=opt,
            data_loader=train_loader,
            device=ddp_gpu,
            epoch=epoch,
            scaler=scaler,
            args=args
        )

        # update scheduler 
        if args.warmup:
            warmup(None)
        else:
            scheduler.step()

        # eval 
        
        sample = evaluate(diffusion, device, args)

        # eval
        # val_loss, val_acc = evaluate(
        #     model=model, 
        #     data_loader=val_loader,
        #     device=ddp_gpu,
        #     epoch=epoch,
        #     classes=args.n_classes
        # )

        # write info into summarywriter in main worker
        if is_main_worker(ddp_gpu):
            tags = ["train_loss", "lr", "sample"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], opt.param_groups[0]['lr'], epoch)
            tb_writer.add_image('sample', sample, epoch)
            # tb_writer.add_scalar(tags[2], train_acc, epoch)
            # tb_writer.add_scalar(tags[3], val_loss, epoch)
            # tb_writer.add_scalar(tags[4], val_acc, epoch)
            

            # save model every two epoch 
            if (epoch % args.save_frequency == 0 and epoch >= 10):
                save_path = os.path.join(args.model_save_path, "model_{}_{:.3f}_.pth".format(epoch, train_loss))
                torch.save(model, save_path)
            # elif (epoch >= 10 and score < val_acc):
            #     save_path = os.path.join(args.model_save_path, "model_{}_{:.3f}_.pth".format(epoch, val_acc))
            #     torch.save(model, save_path)
            #     score = val_acc

if __name__ == '__main__':

    # get args
    args = create_parser()

    # init random seed
    init(args.seed)

    # train in ddp or not
    if args.use_ddp:
        n_gpus_per_node = torch.cuda.device_count()
        world_size = n_gpus_per_node
        mp.spawn(train_ddp, nprocs=n_gpus_per_node, args=(world_size, args))
    else:
        train(args)

    