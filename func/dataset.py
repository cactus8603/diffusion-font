import torch
import cv2
import json
import os
import torchvision
import random
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage, Normalize, Grayscale
from labml import lab, tracker, experiment, monit
# from labml.configs import BaseConfigs, option
# from labml_helpers.device import DeviceConfigs
# from labml_nn.diffusion.ddpm import DenoiseDiffusion
# from labml_nn.diffusion.ddpm.unet import UNet

# from utils import Configs
# from utils import utils

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class ImgDataSet(Dataset):
    def __init__(self, img_data, img_label, n_classes, dict_path):
        super().__init__()
        self.img_data = img_data
        self.img_label = img_label
        self.num_classes = n_classes
        self.font_class_dict = json.load(open(dict_path))


        self.transform = Compose([
            ToPILImage(),
            Resize((224, 224)), 
            Grayscale(num_output_channels = 1),
            ToTensor(),
            Normalize([0.5], [0.1]),
        ])

    def __getitem__(self, idx):
        img = cv2.imread(self.img_data[idx])
        label = self.img_label[idx]
        # print(label, label)
        

        img_tensor = self.transform(img).contiguous()
        label_tensor = torch.zeros(self.num_classes)
        font_num = int(list(self.font_class_dict.keys())[list(self.font_class_dict.values()).index(label)])
        label_tensor[font_num] = 1.

        return img_tensor, label_tensor
    
    def __len__(self):
        return len(self.img_data)
    
class StyleDataSet(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.styles = os.listdir(self.root_dir)

        self.images_per_style = []  # 儲存每種風格的圖片路徑列表
        for style in self.styles:
            style_folder = os.path.join(root_dir, style)
            images = sorted([os.path.join(style_folder, img) for img in os.listdir(style_folder)])
            self.images_per_style.append(images)

        self.transform = Compose([
            ToPILImage(),
            # Resize((224, 224)), 
            Resize((224, 224)), 
            Grayscale(num_output_channels = 1),
            ToTensor(),
            Normalize([0.5], [0.1]),
        ])

        self.transform_3chans = Compose([
            ToPILImage(),
            Resize((224, 224)), 
            Grayscale(num_output_channels = 1),
            ToTensor(),
            Normalize([0.5], [0.1]),
        ])

    def __getitem__(self, idx):
        # style_idx = idx % len(self.styles)
        # style_folder = os.path.join(self.root_dir, self.styles[style_idx])
        # images = os.listdir(style_folder)
        # img_idx = idx // len(self.styles) % len(images)
        # image_path = os.path.join(style_folder, images[img_idx])

        cv2.setNumThreads(0)

        # print(len(self.images_per_style[0]))
        num_styles = len(self.styles)


        # style_idx_1 = idx // (num_styles) - 1  # 第一種風格的索引
        # style_idx_2 = idx % (num_styles) - 1   # 第二種風格的索引
        # random_style_idx = random.randint(0, 49)
        style_idx_1 = idx % num_styles # 第一种风格的索引
        style_idx_2 = (idx + style_idx_1 + random.randint(0, 49)) % num_styles  # 第二种风格的索引，确保不是同一种风格

        # 如果 style_idx_2 与 style_idx_1 相同，循环直到它们不相同为止
        while style_idx_2 == style_idx_1:
            style_idx_2 = (style_idx_2 + 1) % num_styles

        # print(style_idx_1, style_idx_2)

        # image_idx_1 = (style_idx_1 * (len(self.images_per_style[style_idx_1]) - 1)) % len(self.images_per_style[style_idx_1])
        # image_idx_2 = (style_idx_2 * (len(self.images_per_style[style_idx_2]) - 1)) % len(self.images_per_style[style_idx_2])

        # print(idx % len(self.images_per_style[style_idx_1]) - 1)
        image_idx_1 = idx % len(self.images_per_style[style_idx_1]) 
        image_idx_2 = idx % len(self.images_per_style[style_idx_2]) 
        

        # print(style_idx_1, image_idx_1)
        image_path_1 = self.images_per_style[style_idx_1][image_idx_1]
        image_path_2 = self.images_per_style[style_idx_2][image_idx_2]

        image_1 = cv2.imread(image_path_1)
        image_2 = cv2.imread(image_path_2)

        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
            # print('image:', image_2.shape)

        return image_1, image_2, style_idx_1, style_idx_2

        
        # 读取图像
        image = cv2.imread(image_path)

        if self.transform:
            image = self.transform(image)

        return image, style_idx  # 返回图像和对应的风格索引
    
    def __len__(self):

        return sum(len(os.listdir(os.path.join(self.root_dir, style))) for style in self.styles)
    
class MNISTDataset(torchvision.datasets.MNIST):
    """
    ### MNIST dataset
    """

    def __init__(self, image_size):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

        super().__init__(str(lab.get_data_path()), train=True, download=True, transform=transform)

    def __getitem__(self, item):
        return super().__getitem__(item)[0]
    

    
