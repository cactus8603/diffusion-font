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
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage, Normalize, Grayscale

# from typing import List
# from labml import lab, tracker, experiment, monit
# from labml_helpers.seed import SeedConfigs
# from labml.configs import BaseConfigs, option
# from labml_helpers.device import DeviceConfigs
# from labml_nn.diffusion.ddpm import DenoiseDiffusion
# from labml_helpers.train_valid import BatchIndex
# from labml_nn.diffusion.ddpm.unet import UNet

from func.dataset import ImgDataSet, StyleDataSet # , MNISTDataset
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
    # train_style_dataset = StyleDataSet(args.style_dir)
    
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

def get_style_loader(args):
    train_style_dataset = StyleDataSet(args.data_path)

    # dist
    train_sampler = DistributedSampler(train_style_dataset)
    
    train_loader = DataLoader(
        train_style_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        sampler=train_sampler # dist
    )

    return train_loader


# def load_model(device, args):
    
#     model = UNet(
#         image_channels=args.image_channels,
#         n_channels=args.n_channels,
#         ch_mults=args.ch_mults,
#         is_attn=args.is_attn
#     ).to(device)

#     print("load model successful")

#     return model

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def total_variation_loss(img):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
    tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
    return (tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

def train_one_epoch(model, style_encoder, content_encoder, optimizer, data_loader, device, epoch, scaler, args):
    model.train()
    # loss_function = torch.nn.CrossEntropyLoss()
    loss_function = torch.nn.MSELoss()

    # accu_loss = torch.zeros(1).to(device)
    # avg_loss = torch.zeros(1).to(device)
    # accu_num = torch.zeros(1).to(device)

    optimizer.zero_grad()

    # sample_num = 0
    pbar = tqdm(data_loader)
    # image_1, image_2, style_idx_1, style_idx_2
    for i, (img_1, img_2, style_idx_1, style_idx_2) in enumerate(data_loader):
        img_1, img_2 = img_1.to(device), img_2.to(device)
        # img, label = img.to(device), label.to(device)
        style_feature_map = style_encoder.forward_features(img_2)
        content_feature_map = content_encoder.forward_features(img_1)
        # print(style_feature_map.shape)
        # style_feature_map = style_encoder(img_2)
        # img_1 = F.interpolate(img_1, size=(args.image_size, args.image_size), mode='bilinear', align_corners=False)
        # upsampled_style_feature_map = F.interpolate(style_feature_map, size=(28, 28), mode='bilinear', align_corners=False)

        upsampled_tensor = style_feature_map.view(-1, 1, 176, 49)
        # print(upsampled_tensor.shape)
        upsampled_tensor = torch.nn.functional.interpolate(upsampled_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        # upsampled_tensor = upsampled_tensor.squeeze(0).squeeze(0)
        # print(upsampled_tensor.shape)
        # print(img_1.shape)
        
        # combined_tensor = torch.cat((img_1, upsampled_style_feature_map), dim=1)


        # print(upsampled_style_feature_map.shape)
        # feature map
        # 1.(batch, 32, 56, 56)
        # 2.(batch, 48, 28, 28)
        # 3.(batch, 96, 14, 14)
        # 4.(batch, 176, 7, 7)
        # Unet bottom (512, 28, 28)

        # style_feature_map = style_encoder(img)
        # for _ in style_feature_map:
        #     print(_.shape)
        # break
        # sample_num += img.shape[0]

        with autocast():
            # loss = diffusion.loss(img)
            # generate_img = model(img, style_feature_map)
            # mse_loss, img_pred = model(combined_tensor)
            mse_loss, img_pred = model(img_1, upsampled_tensor)
            # print(img_pred.shape)
            pred_style_feature = style_encoder.forward_features(img_pred)
            style_loss = F.mse_loss(style_feature_map, pred_style_feature)

            pred_content_feature = content_encoder.forward_features(img_pred)
            content_loss = F.mse_loss(content_feature_map, pred_content_feature)

            tv_loss = total_variation_loss(img_pred)

            loss = mse_loss*0.01 + content_loss + style_loss 
            # loss = mse_loss + content_loss

            # loss = style_loss + content_loss # + tv_loss

            # print(style_loss)

        # accu_loss += loss.detach()
        loss /= args.accumulation_step
        scaler.scale(loss).backward(retain_graph=True)

        if (((i+1) % args.accumulation_step == 0) or (i+1 == len(data_loader))):
            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()
            optimizer.zero_grad()
   
        pbar.desc = "epoch:{},c_loss:{:.3f}, s_loss:{:.3f}, tv_loss:{:.3f} loss:{:.3f}".format(epoch, content_loss.item(), style_loss.item(), tv_loss.item(), loss.item())
        pbar.update(1)
        # break

    return mse_loss.item(), style_loss.item(), content_loss.item(), loss.item()

@torch.no_grad()
def style_evaluate(model, diffusion, style_encoder, epoch, device, args):
    model.eval()

    #  = torch.load(args.style_enc, map_location=device)
    
    # style_feature_map = style_encoder.forward_features(img_tensor)

    transform = Compose([
        ToPILImage(),
        Resize((224, 224)), 
        Grayscale(num_output_channels = 1),
        ToTensor(),
        Normalize([0.5], [0.1]),
    ])

    x = []
    sample_set = glob(os.path.join(args.sample_set, '*.png'))
    img_set = ['content.png', 'style_1.png', 'style_2.png', 'style_3.png']

    img_content = cv2.imread(os.path.join(args.sample_set, 'content.png'))
    x = transform(img_content).contiguous().unsqueeze(0).to(device)

    with torch.no_grad():
        b, c, h, w, device, img_size, = *x.shape, x.device, args.image_size
        t = torch.randint(0, args.n_steps, (b,), device=device).long().to(device)
        noise = None
        noise = default(noise, lambda: torch.randn_like(x)).to(device)

        for img in img_set:
            img_style = cv2.imread(os.path.join(args.sample_set, img))
            img_style = transform(img_style).contiguous().unsqueeze(0).to(device)

            style_feature_map = style_encoder.forward_features(img_style)
            upsampled_tensor = style_feature_map.view(-1, 1, 176, 49)
            upsampled_tensor = torch.nn.functional.interpolate(upsampled_tensor, size=(224, 224), mode='bilinear', align_corners=False)

            # style_feature_map = style_encoder(img_style)
            # upsampled_style_feature_map = F.interpolate(style_feature_map[2], size=(args.image_size, args.image_size), mode='bilinear', align_corners=False)

            # combined_tensor = torch.cat((x, upsampled_style_feature_map), dim=1)

            # x_noisy = diffusion.q_sample(x_start=combined_tensor, t=t, noise=noise)
            # x_recon = model.forward(x_noisy, t)
            # x_recon = diffusion.unnormalize(x_recon)

            # mse_loss, img_pred = diffusion(combined_tensor)
            mse_loss, img_pred = diffusion(x, upsampled_tensor)

            if device == torch.device('cuda', 0):
                torchvision.utils.save_image(img_pred, os.path.join(args.model_save_path, '{}_{}'.format(epoch, img)))
    if device == torch.device('cuda', 0):
        print('save grid sample at epoch {}'.format(epoch))

@torch.no_grad()
def evaluate(model, diffusion, epoch, device, args):

    model.eval()

    #  = torch.load(args.style_enc, map_location=device)
    
    # style_feature_map = style_encoder.forward_features(img_tensor)

    transform = Compose([
        ToPILImage(),
        Resize((128, 128)), 
        Grayscale(num_output_channels = 1),
        ToTensor(),
        Normalize([0.5], [0.1]),
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
        x_noisy = diffusion.q_sample(x_start=x, t=t, noise=noise)
        # # print(x_noisy.shape, t.shape)

        # x_recon = model.module.forward(x_noisy, t)
        x_recon = model.forward(x_noisy, t)
        # # print(x_recon)
        x_recon = diffusion.unnormalize(x_recon)
        # x_recon = (1 - x_recon)

        grid = torchvision.utils.make_grid(x_recon, nrow=4)
        if device == torch.device('cuda', 0):
            torchvision.utils.save_image(grid, os.path.join(args.model_save_path, 'recon_{}.png'.format(epoch)))
            print('save grid sample at epoch {}'.format(epoch))






        
        

        