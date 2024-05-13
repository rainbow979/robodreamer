from guided_diffusion.guided_diffusion.unet import UNetModel
from torch import nn
import torch.nn.functional as F
import torch
from einops import repeat, rearrange
from config import config
from torch.distributions import Bernoulli
from diffusers import DiffusionPipeline
import os
import numpy as np

class Unet(nn.Module):
    def __init__(self, image_size=128):
        super(Unet, self).__init__()
        if config.config['latent']:
            self.base_channel = 4
        else:
            self.base_channel = 3
        self.unet = UNetModel(
            image_size=image_size,
            in_channels=self.base_channel*2 if config.config['tiling'] else self.base_channel,
            model_channels=config.config['model_channels'],
            out_channels=self.base_channel,
            num_res_blocks=config.config['num_res_blocks'],
            attention_resolutions=config.config['attention_resolutions'],
            dropout=0,
            channel_mult=config.config['channel_mult'],
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=4096 if config.config['text'] == 'T5' else 512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
        )
        self.mask_dist = Bernoulli(probs=1 - 0.4)
        

    def forward(self, x, x_cond, t, task_embed=None, use_dropout=True, force_dropout=False, **kwargs):
        c = self.base_channel
        f = x.shape[1] // c
        
        x = rearrange(x, 'b (f c) h w -> b c f h w', c=c)

        if config.config['tiling']:
            x_cond = repeat(x_cond, 'b c h w -> b c f h w', f=f)
            x = torch.cat([x_cond, x], dim=1)
        else:
            x_cond = x_cond[:, :, None]
            x = torch.cat([x_cond, x], dim=2)

        
        out = self.unet(x, t, task_embed, **kwargs)
        if not config.config['tiling']:
            out = out[:, :, 1:]

        return rearrange(out, 'b c f h w -> b (f c) h w')


      

class SSR(nn.Module):
    def __init__(self, image_size=128):
        super(SSR, self).__init__()
        if config.config['latent']:
            self.base_channel = 4
        else:
            self.base_channel = 3
        self.unet = UNetModel(
            image_size=image_size,
            in_channels=self.base_channel*2,
            model_channels=config.config['model_channels'],
            out_channels=self.base_channel,
            num_res_blocks=config.config['num_res_blocks'],
            attention_resolutions=config.config['attention_resolutions'],
            dropout=0,
            channel_mult=config.config['channel_mult'],
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
        )
        self.mask_dist = Bernoulli(probs=1 - 0.2)

    def forward(self, x, x_cond, t, task_embed=None, use_dropout=True, force_dropout=False, low_res=None, **kwargs):
        c = self.base_channel
        f = x.shape[1] // c - 1
        b = x.shape[0]
        x = rearrange(x, 'b (f c) h w -> b c f h w', c=c)
        h, w = x.shape[-2:]
        x_cond = rearrange(x_cond, 'b (f c) h w -> (b f) c h w', c=c)
        x_cond = F.interpolate(x_cond, size=(h, w), mode='bilinear')
        x_cond = rearrange(x_cond, '(b f) c h w -> b c f h w', b=b)
        x = torch.cat([x_cond, x], dim=1)
        
        out = self.unet(x, t, task_embed, **kwargs)
        
        return rearrange(out, 'b c f h w -> b (f c) h w')