import torch
import torch.nn as nn

from labml_nn.diffusion.ddpm.unet import UNet
from labml_nn.diffusion.ddpm import DenoiseDiffusion
from labml import lab, tracker, experiment, monit

class Diffusion(nn.Module):
    def __init__(self, device, args):
        super().__init__()
        
        # parameters
        self.image_channels = args.image_channels
        self.image_size = args.image_size
        self.n_channels = args.n_channels
        self.ch_mults = [1, 2, 2, 4]
        self.is_attn = args.is_attn
        self.n_steps = args.n_steps
        self.device = device

        self.eps_model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.ch_mults,
            is_attn=self.is_attn
        )

        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device=self.device
        )

        # Image logging
        # tracker.set_image("sample", True)

    def forward(self, x):
        return self.diffusion.loss(x)

    def sample(self):
        with torch.no_grad():
            x = torch.rand([self.n_sample, self.image_channels, self.image_size, self.image_size],
                           device=self.device)
            
            for t_ in monit.iterate('Sample', self.n_steps):
                print(t_)
                # t = self.n_steps - t_ -1
                # x = self.diffusion.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))

            # tracker.save('sample', x)
            return x