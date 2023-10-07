import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from torch.utils.data import DataLoader
from torch.cuda.amp import autocast as autocast
from torch.cuda import amp
# from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

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


def train_one_epoch(model, diffusion, optimizer, data_loader, device, epoch, scaler, args):
    model.train()
    # loss_function = torch.nn.CrossEntropyLoss()

    accu_loss = torch.zeros(1).to(device)
    # avg_loss = torch.zeros(1).to(device)
    # accu_num = torch.zeros(1).to(device)

    optimizer.zero_grad()

    sample_num = 0
    pbar = tqdm(data_loader)

    for i, (img, label) in enumerate(data_loader):
        img, label = img.to(device), label.to(device)

        sample_num += img.shape[0]

        with autocast():
            loss = diffusion.loss(img)
            # loss = loss_function(pred, label)

        accu_loss += loss.detach()
        loss /= args.accumulation_step
        scaler.scale(loss).backward()

        if (((i+1) % args.accumulation_step == 0) or (i+1 == len(data_loader))):
            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()
            optimizer.zero_grad()
   
        pbar.desc = "epoch:{},gpu:{},loss:{:.5f}".format(epoch, device, accu_loss.item()/(i+1))
        pbar.update(1)
        # break

    return accu_loss.item() / (i+1)

@torch.no_grad()
def evaluate(diffusion, device, args):
    with torch.no_grad():
        x = torch.rand([args.n_samples, args.image_channels, args.image_size, args.image_size],
                        device=device)
        
        for t_ in range(args.n_steps):
            # print(t_)
            t = args.n_steps - t_ -1
            x = diffusion.p_sample(x, x.new_full((args.n_samples,), t, dtype=torch.long))

        # tracker.save('sample', x)
        return x.squeeze(0)

# @torch.no_grad()
# def evaluate(model, data_loader):
#     model.eval()
#     loss_function = torch.nn.CrossEntropyLoss()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     accu_loss = torch.zeros(1).to(device)
#     accu_num = torch.zeros(1).to(device)

#     sample_num = 0
#     data_loader = tqdm(data_loader)

#     for i, (img, label) in enumerate(data_loader):
#         img, label = img.to(device), label.to(device)
#         sample_num += img.shape[0]

#         pred = model(img)
        
#         p = F.softmax(pred, dim=1)
#         accu_num += (p.argmax(1) == label.argmax(1)).type(torch.float).sum().item()

#         loss = loss_function(pred, label)
#         accu_loss += loss
        
#         data_loader.desc = "loss:{:.5f}, acc:{:.5f}".format(accu_loss.item()/(i+1), accu_num.item() / sample_num)
    
#     return accu_num.item() / sample_num
