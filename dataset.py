import torch
import cv2
import json
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage, Normalize
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
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, idx):
        img = cv2.imread(self.img_data[idx])
        label = self.img_label[idx]
        # print(label, label)

        img_tensor = self.transform(img)
        label_tensor = torch.zeros(self.num_classes)
        font_num = int(list(self.font_class_dict.keys())[list(self.font_class_dict.values()).index(label)])
        label_tensor[font_num] = 1.

        return img_tensor, label_tensor
    
    def __len__(self):
        return len(self.img_data)
    
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
    
