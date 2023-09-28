import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import timm
import pathlib

from timm.models.efficientformer_v2 import _cfg, efficientformerv2_s0

import os
import random
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from .dataset import ImgDataSet

def read_spilt_data(args):

    # for not spilt train and val
    assert os.path.exists(args.data_path), "data path:{} does not exist".format(args.data_path)

    font_class = glob(os.path.join(args.data_path, '*'))
    font_class.sort()

    val_data = []
    val_label = []

    for cla in font_class:
        img = glob(os.path.join(cla, '*'))
        # img_class = font_class_indices[cla]
        img_font = os.path.basename(cla).split('.')[0]

        for img_path in img:
            val_data.append(img_path)
            val_label.append(img_font)

    return val_data, val_label


def get_loader(args):
    val_data, val_label = read_spilt_data(args)

    val_dataset = ImgDataSet(val_data, val_label, args.n_classes, args.dict_path)
    
    val_sampler = DistributedSampler(val_dataset)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        sampler=val_sampler
    )

    return val_loader

def load_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dist.init_process_group(
        backend='nccl',
        init_method="tcp://127.0.0.1:" + str(args.port),
        world_size=1,
        rank=0,
    )

    model = torch.load(args.model_path, map_location=device)  
    print("load model successful")

    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler, args_dict):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()

    accu_loss = torch.zeros(1).to(device)
    avg_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)

    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)

    for i, (img, label) in enumerate(data_loader):
        img, label = img.to(device), label.to(device)
        # print(img.shape)
        # print(label.argmax(1))
        # print(label.shape)
        # break
        sample_num += img.shape[0]

        with autocast():
            pred = model(img)
            loss = loss_function(pred, label)

        accu_loss += loss.detach()
        loss /= args_dict['accumulation_step']
        scaler.scale(loss).backward()

        p = F.softmax(pred, dim=1)
        accu_num += (p.argmax(1) == label.argmax(1)).type(torch.float).sum().item()

        # pred_class = torch.max(pred, dim=1)[1]
        # accu_num += torch.eq(pred_class, label).sum()

        if (((i+1) % args_dict['accumulation_step'] == 0) or (i+1 == len(data_loader))):
            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()
            optimizer.zero_grad()

        # print(accu_loss.item(), loss.detach(), loss.item())     
        data_loader.desc = "epoch:{},gpu:{},loss:{:.5f},acc:{:.5f}".format(epoch, device, accu_loss.item()/(i+1), accu_num.item() / sample_num)
        # break

    return (accu_loss.item() / (i+1)), (accu_num.item() / sample_num)

@torch.no_grad()
def evaluate(model, data_loader):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader)

    for i, (img, label) in enumerate(data_loader):
        img, label = img.to(device), label.to(device)
        sample_num += img.shape[0]

        pred = model(img)
        
        p = F.softmax(pred, dim=1)
        accu_num += (p.argmax(1) == label.argmax(1)).type(torch.float).sum().item()

        loss = loss_function(pred, label)
        accu_loss += loss
        
        data_loader.desc = "loss:{:.5f}, acc:{:.5f}".format(accu_loss.item()/(i+1), accu_num.item() / sample_num)
    
    return accu_num.item() / sample_num
