import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
# import torch.distributed as dist
# import timm
# import pathlib
# import torchvision

# from timm.models.efficientformer_v2 import _cfg, efficientformerv2_s0

import os
# import random
import torch
# import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
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
    train_data, train_label = read_spilt_data(args)

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

def load_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dist.init_process_group(
    #     backend='nccl',
    #     init_method="tcp://127.0.0.1:" + str(args.port),
    #     world_size=1,
    #     rank=0,
    # )

    model = torch.load(args.model_path, map_location=device)  
    print("load model successful")

    return model


class Configs(BaseConfigs):
    """
    ## Configurations
    """
    # Device to train the model on.
    # [`DeviceConfigs`](https://docs.labml.ai/api/helpers.html#labml_helpers.device.DeviceConfigs)
    #  picks up an available CUDA device or defaults to CPU.
    device: torch.device = DeviceConfigs()

    set_seed = SeedConfigs()

    # U-Net model for $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
    model: UNet
    # [DDPM algorithm](index.html)
    diffusion: DenoiseDiffusion

    # Number of channels in the image. $3$ for RGB.
    image_channels: int = 3
    # Image size
    image_size: int = 224
    # Number of channels in the initial feature map
    n_channels: int = 32
    # The list of channel numbers at each resolution.
    # The number of channels is `channel_multipliers[i] * n_channels`
    channel_multipliers: List[int] = [1, 2, 2, 4]
    # The list of booleans that indicate whether to use attention at each resolution
    is_attention: List[int] = [False, False, False, True]

    # Number of time steps $T$
    n_steps: int = 10
    # Batch size
    batch_size: int = 8
    # Number of samples to generate
    n_samples: int = 4
    # Learning rate
    learning_rate: float = 2e-4

    # Number of training epochs
    epochs: int = 200

    # Dataset
    dataset: torch.utils.data.Dataset
    # Sampler
    sampler: torch.utils.data.distributed.DistributedSampler
    # Dataloader
    data_loader: torch.utils.data.DataLoader

    # Adam optimizer
    optimizer: torch.optim.Adam


    def init(self):
        # Create $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ model
        self.model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
        ).to(self.device)

        # Create [DDPM class](index.html)
        self.diffusion = DenoiseDiffusion(
            model=self.model,
            n_steps=self.n_steps,
            device=self.device,
        )

        # Create dataloader
        self.sampler = DistributedSampler(self.dataset, shuffle=True)
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset, 
            self.batch_size, 
            shuffle=True, 
            pin_memory=True,
            sampler=self.sampler
        )
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Image logging
        tracker.set_image("sample", True)

    def sample(self):
        """
        ### Sample images
        """
        with torch.no_grad():
            # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
            x = torch.randn([self.n_samples, self.image_channels, self.image_size, self.image_size],
                            device=self.device)

            # Remove noise for $T$ steps
            for t_ in monit.iterate('Sample', self.n_steps):
                # $t$
                t = self.n_steps - t_ - 1
                # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
                x = self.diffusion.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))

            # Log samples
            tracker.save('sample', x)

    def train(self):
        """
        ### Train
        """
        # print(type(self.data_loader))
        # Iterate through the dataset
        # for (data, label) in monit.iterate('Train', self.data_loader):
            # Increment global step
            # tracker.add_global_step()
            # # Move data to device
            # data = data.to(self.device)

            # # Make the gradients zero
            # self.optimizer.zero_grad()
            # # Calculate loss
            # loss = self.diffusion.loss(data)
            # # Compute gradients
            # loss.backward()
            # # Take an optimization step
            # self.optimizer.step()
            # # Track the loss
            # tracker.save('loss', loss)

        for batch_idx, (data, label) in monit.enum("Train", self.data_loader):
            
            # Move data to device
            data = data.to(self.device)
            # Make the gradients zero
            self.optimizer.zero_grad()
            # Calculate loss
            loss = self.diffusion.loss(data)
            # Compute gradients
            loss.backward()
            # Take an optimization step
            self.optimizer.step()
            # Increment global step
            tracker.add_global_step(data.shape[0])
            # Track the loss
            tracker.save('loss', loss)


    def step(self, batch: any, batch_idx: BatchIndex):
        data, label = batch[0].to(self.device), batch[1].to(self.device)

        tracker.add_global_step(len(data))

        # Make the gradients zero
        self.optimizer.zero_grad()
        # Calculate loss
        loss = self.diffusion.loss(data)
        # Compute gradients
        loss.backward()
        # Take an optimization step
        self.optimizer.step()
        # Track the loss
        tracker.save('loss', loss)


    def run(self):
        """
        ### Training loop
        """
        for _ in monit.loop(self.epochs):
            # Train the model
            self.train()
            # self.step()
            # Sample some images
            self.sample()
            # New line in the console
            tracker.new_line()
            # Save the model
            experiment.save_checkpoint()




# @option(Configs.dataset, 'MNIST')
# def mnist_dataset(c: Configs):
#     """
#     Create MNIST dataset
#     """
#     return MNISTDataset(c.image_size)


@option(Configs.dataset, 'ImgDataset')
def Img_dataset(c: Configs):
    """
    Create CelebA dataset
    """
    from train_diff import create_parser
    args = create_parser()
    train_data, train_label = read_spilt_data(args.data_path)

    return ImgDataSet(train_data, train_label, args.n_classes, args.font_classes)

@option(Configs.model)
def ddp_model(c: Configs):
    return DistributedDataParallel(UNet().to(c.device), device_ids=[c.device])



def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler, args):
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
            loss = model(img)
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
        # break

    return accu_loss.item() / (i+1)

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
