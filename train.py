import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import random

import torch
import math
import numpy as np
import timm
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.datasets as datasets
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage, Normalize
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torch.optim.lr_scheduler as lr_scheduler
import argparse
from typing import List
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from ignite.handlers import create_lr_scheduler_with_warmup
# from labml_nn.diffusion.ddpm import DenoiseDiffusion
# from labml_nn.diffusion.ddpm.unet import UNet
from tensorboardX import SummaryWriter
# from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
from torch.cuda import amp
# from func.dataset import ImgDataSet
# from model.UNet import UNet
# from model.diffusion import DenoiseDiffusion
from func.utils import get_loader, train_one_epoch, load_model, evaluate
# from model.diff import Diffusion
from model.diff_ai import CustomModel
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, Unet1D, GaussianDiffusion1D



def create_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config_path", default="config.yaml", nargs='?', help="path to config file")
    parser.add_argument("--data_path", default='/code/Font/byFont', type=str, help='')
    parser.add_argument("--sample_set", default='./sample', type=str, help='')
    parser.add_argument("--json_file", default='./cfgs/font_classes_173.json', type=str, help='')

    # ddp setting
    parser.add_argument("--use_ddp", default=True, type=bool, help='use ddp or not')
    parser.add_argument("--port", default=8888, type=int, help='ddp port')

    # training setting
    parser.add_argument("--lr", default=5e-6, type=float, help='learning rate')
    parser.add_argument("--epoch", default=2000, type=int, help='total epoch')
    parser.add_argument("--n_classes", default=173, type=int, help='total classes')
    parser.add_argument("--n_steps", default=1200, type=int, help='')
    parser.add_argument("--n_samples", default=1, type=int, help='Number of samples to generate, only 1 now')
    parser.add_argument("--accumulation_step", default=4, type=int, help='')
    parser.add_argument("--seed", default=8603, type=int, help='init random seed')

    # save and load data path
<<<<<<< HEAD
    parser.add_argument("--model_save_path", default='./result/1_chs', type=str, help='path to save model')
=======
    parser.add_argument("--model_save_path", default='./result/test_eval', type=str, help='path to save model')
>>>>>>> 586a2fbf356d6981355ca31ac0bebcd8df82ae33
    parser.add_argument("--save_frequency", default=5, type=int, help='save model frequency')
    parser.add_argument("--sample_freq", default=2, type=int, help='')
    parser.add_argument("--style_enc", default='./cfgs/style.pt', type=str, help='path to style encoder')

    # images setting
    parser.add_argument("--image_size", default=128, type=int, help='size of input image')
    parser.add_argument("--image_channels", default=3, type=int, help='RGB')
    
    
    # parser.add_argument("--n_channels", default=8, type=int, help='Number of channels in the initial feature map')
    # parser.add_argument("--patch_size", default=4, type=int, help='')
    # parser.add_argument("--is_attn", default=[False, False, False, True], type=List[int], help='')
    # parser.add_argument("--ch_mults", default=[1, 2, 2, 4], type=List[int], help='')

    # warmup
    parser.add_argument("--warmup", default=False, type=bool, help='use warmup or not')
    parser.add_argument("--warmup_start_value", default=5e-10, type=float, help='')
    parser.add_argument("--warmup_step", default=3, type=int, help='warmup steps')

    ### parameter setting ###
    # optim and lr scheduler
    parser.add_argument("--momentum", default=0.937, type=float, help='')
    parser.add_argument("--weight_decay", default=0.00005, type=float, help='')
    parser.add_argument("--lrf", default=0.0005, type=float, help='')
    parser.add_argument("--cosanneal_cycle", default=50, type=int, help='')

<<<<<<< HEAD
    parser.add_argument("--batch_size", default=48, type=int, help='')
=======
    parser.add_argument("--batch_size", default=80, type=int, help='')
>>>>>>> 586a2fbf356d6981355ca31ac0bebcd8df82ae33
    parser.add_argument("--num_workers", default=6, type=int, help='')

    # model setting
    # parser.add_argument("--dropout_prob", default=0.1, type=float, help='')
    # parser.add_argument("--max_positional_encoding", default=10000, type=int, help='')
    # parser.add_argument("--num_heads", default=3, type=int, help='')

    # parser.add_argument("", default=, type=, help='')
    
    args = parser.parse_args()
    return args

def is_main_worker(gpu):
    return (gpu <= 0)

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


    # style_extracter = torch.load(args.style_enc, map_location=device)
    # style_extracter = timm.create_model('efficientformerv2_s0', features_only=True, pretrained=True).to(device)
    style_extracter = 0
    # print("Load style encoder and model successfully")


    # diffusion model
<<<<<<< HEAD
    # model = Unet(
    #     dim = 32,
    #     # self_condition = True,
    #     # learned_sinusoidal_cond = True,
    #     # random_fourier_features = True,
    #     dim_mults = (1, 2, 4, 8),
    #     flash_attn = True,
    # )

    # model = torch.load('/code/diffusion-font/result/train_v2_adamW/model_85_0.003_.pth', map_location=device)
    # diffusion = GaussianDiffusion(
    #     model,
    #     image_size = args.image_size,
    #     timesteps = args.n_steps,    # number of steps
    #     # sampling_timesteps = 250
    #     # objective = 'pred_x0'
    # ).to(device)

    # for 1 channels
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 1
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = args.image_size,
        timesteps = args.n_steps,   # number of steps
        # sampling_timesteps = 10
=======
    model = Unet(
        dim = 32,
        # self_condition = True,
        # learned_sinusoidal_cond = True,
        # random_fourier_features = True,
        dim_mults = (1, 2, 4, 8),
        flash_attn = True,
    )

    model = torch.load('/code/diffusion-font/result/train_v2_adamW/model_85_0.003_.pth', map_location=device)
    diffusion = GaussianDiffusion(
        model,
        image_size = args.image_size,
        timesteps = args.n_steps,    # number of steps
        # sampling_timesteps = 250
        # objective = 'pred_x0'
>>>>>>> 586a2fbf356d6981355ca31ac0bebcd8df82ae33
    ).to(device)
    print("load model successful")


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
    
    # score = 0
    
    # setting Automatic mixed precision
    scaler = amp.GradScaler()

    # start training
    for epoch in range(start_epoch, args.epoch):
<<<<<<< HEAD

        # train 
        train_loss = train_one_epoch(
            model=diffusion, 
            style_extracter=style_extracter,
            optimizer=opt,
            data_loader=train_loader,
            device=ddp_gpu,
            epoch=epoch,
            scaler=scaler,
            args=args
        )

=======

        # train 
        # train_loss = train_one_epoch(
        #     model=diffusion, 
        #     style_extracter=style_extracter,
        #     optimizer=opt,
        #     data_loader=train_loader,
        #     device=ddp_gpu,
        #     epoch=epoch,
        #     scaler=scaler,
        #     args=args
        # )

>>>>>>> 586a2fbf356d6981355ca31ac0bebcd8df82ae33
        # # update scheduler 
        if args.warmup:
            warmup(None)
        else:
            scheduler.step()

        # eval
        evaluate(
            model=model,
            diffusion=diffusion, 
            # data_loader=test_loader,
            device=ddp_gpu,
            epoch=epoch,
            # classes=args.n_classes,
            args=args, 
        )

<<<<<<< HEAD
        # break
=======
        break
>>>>>>> 586a2fbf356d6981355ca31ac0bebcd8df82ae33

        # write info into summarywriter in main worker
        if is_main_worker(ddp_gpu):
            tags = ["train_loss", "lr", "sample_0", "sample_1", "sample_2", "sample_3"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], opt.param_groups[0]['lr'], epoch)


            # save model every two epoch 
            if (epoch % args.save_frequency == 0 and epoch >= 10):
                save_path = os.path.join(args.model_save_path, "model_{}_{:.3f}_.pth".format(epoch, train_loss))
                torch.save(model.module, save_path)


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

    