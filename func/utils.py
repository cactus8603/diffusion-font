import os
import cv2

import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from glob import glob
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from torch.utils.data import DataLoader
from torch.cuda.amp import autocast as autocast
from torch.cuda import amp
from inspect import isfunction
import torchvision.transforms as T
from PIL import Image  
# from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage, Normalize

from typing import List
from labml import lab, tracker, experiment, monit
from labml_helpers.seed import SeedConfigs
from labml.configs import BaseConfigs, option
from labml_helpers.device import DeviceConfigs
from labml_nn.diffusion.ddpm import DenoiseDiffusion
from labml_helpers.train_valid import BatchIndex
from labml_nn.diffusion.ddpm.unet import UNet

from func.dataset import ImgDataSet # , MNISTDataset
# from . import dataset



def read_spilt_data(path):

    # for not spilt train and val
    assert os.path.exists(path), "data path:{} does not exist".format(path)

    font_class = glob(os.path.join(path, '*'))
    font_class.sort()

    train_data = []
    train_label = []

    for cla in font_class:
        img = glob(os.path.join(cla, '*'))
        # img_class = font_class_indices[cla]
        img_font = os.path.basename(cla).split('.')[0]

        for img_path in img:
            train_data.append(img_path)
            train_label.append(img_font)

    return train_data, train_label


def get_loader(args):
    train_data, train_label = read_spilt_data(args.data_path)

    train_dataset = ImgDataSet(train_data, train_label, args.n_classes, args.json_file)
    
    # dist
    train_sampler = DistributedSampler(train_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        sampler=train_sampler # dist
    )

    return train_loader

def load_model(device, args):
    
    model = UNet(
        image_channels=args.image_channels,
        n_channels=args.n_channels,
        ch_mults=args.ch_mults,
        is_attn=args.is_attn
    ).to(device)

    print("load model successful")

    return model

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def train_one_epoch(model, style_extracter, optimizer, data_loader, device, epoch, scaler, args):
    model.train()
    # loss_function = torch.nn.CrossEntropyLoss()
    loss_function = torch.nn.MSELoss()

    # accu_loss = torch.zeros(1).to(device)
    # avg_loss = torch.zeros(1).to(device)
    # accu_num = torch.zeros(1).to(device)

    optimizer.zero_grad()

    # sample_num = 0
    pbar = tqdm(data_loader)

    for i, (img, label) in enumerate(data_loader):
        img, label = img.to(device), label.to(device)
        # feature_map = style_extracter.forward_features(img).to(device)
        # feature map
        # 1.(batch, 32, 56, 56)
        # 2.(batch, 48, 28, 28)
        # 3.(batch, 96, 14, 14)
        # 4.(batch, 176, 7, 7)

        # feature_map = style_extracter(img)
        # for _ in feature_map:
        #     print(_.shape)
        # break
        # sample_num += img.shape[0]


        with autocast():
            # loss = diffusion.loss(img)
            # generate_img = model(img, feature_map)
            loss = model(img)

        # accu_loss += loss.detach()
        loss /= args.accumulation_step
        scaler.scale(loss).backward()

        if (((i+1) % args.accumulation_step == 0) or (i+1 == len(data_loader))):
            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()
            optimizer.zero_grad()
   
        pbar.desc = "epoch:{},gpu:{},loss:{:.5f}".format(epoch, device, loss.item())
        pbar.update(1)
        # break

    return loss.item()


@torch.no_grad()
def evaluate(model, diffusion, epoch, device, args):

    model.eval()

    #  = torch.load(args.style_enc, map_location=device)
    
    # feature_map = style_extracter.forward_features(img_tensor)

    transform = Compose([
        ToPILImage(),
        Resize((args.image_size, args.image_size)), 
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    x = []
    sample_set = glob(os.path.join(args.sample_set, '*.png'))
    # print(sample_set)
    for sample in sample_set:
        img = cv2.imread(sample)
        img_tensor = transform(img).contiguous().to(device)
        x.append(img_tensor)

    x = torch.stack(x)

    # print(x)

    with torch.no_grad():
        b, c, h, w, device, img_size, = *x.shape, x.device, args.image_size
        t = torch.randint(0, args.n_steps, (b,), device=device).long().to(device)
        noise = None
        noise = default(noise, lambda: torch.randn_like(x)).to(device)
        # x_noisy = diffusion.q_sample(x_start=x, t=t, noise=noise)
        # # print(x_noisy.shape, t.shape)

        # x_recon = model.module.forward(x_noisy, t)
        # # print(x_recon)
        # x_recon = diffusion.unnormalize(x_recon)
        # x_recon = (1 - x_recon)

        x_recon = diffusion.p_sample_loop((b, 3, args.image_size, args.image_size), return_all_timesteps = False)
        grid = torchvision.utils.make_grid(x_recon, nrow=4)
        grid = torchvision.utils.make_grid(x_recon, nrow=4)
        if device == torch.device('cuda', 0):
            torchvision.utils.save_image(grid, os.path.join(args.model_save_path, 'recon_{}.png'.format(epoch)))
            print('save grid sample at epoch {}'.format(epoch))






        
        

        