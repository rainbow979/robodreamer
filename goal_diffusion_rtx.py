import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from torch.optim import Adam

import torch.distributions as dist

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

#from accelerate import Accelerator

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
import matplotlib.pyplot as plt
import numpy as np

import wandb

from config import config, init_config

__version__ = "0.0"
use_accelerate = False
import os

from pynvml import *
from validation import get_bound_box_labels, draw_bbox

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
    logging.info(f"GPU memory occupied: {info.used//1024**2} MB.")

#import tensorboard as tb

# constants
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions
def tensors2vectors(tensors):
    def tensor2vector(tensor):
        flo = (tensor.permute(1, 2, 0).numpy()-0.5)*1000
        r = 8
        plt.quiver(flo[::-r, ::r, 0], -flo[::-r, ::r, 1], color='r', scale=r*20)
        plt.savefig('temp.jpg')
        plt.clf()
        return plt.imread('temp.jpg').transpose(2, 0, 1)
    return torch.from_numpy(np.array([tensor2vector(tensor) for tensor in tensors])) / 255

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

import pickle
   
class GoalGaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels=3,
        timesteps = 1000,
        sampling_timesteps = 100,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        wt = 8,
    ):
        super().__init__()
        # assert not (type(self) == GoalGaussianDiffusion and model.channels != model.out_dim)
        # assert not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = channels

        self.image_size = image_size

        self.objective = objective

        self.wt = wt

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_cond, task_embed,  clip_x_start=False, rederive_pred_noise=False):
        
        #model_output = self.model(torch.cat([x_cond, x], dim=1), t, task_embed)
        #print(task_embed[0].shape, len(task_embed))
        if type(task_embed) == list and len(task_embed) > 0:
            uncond_eps = self.model(x, x_cond, t, None, force_dropout=True)
            '''model_output = []
            for embed in task_embed:
                cond_eps = self.model(x, x_cond, t, embed, use_dropout=False)
                wt = 8
                model_output.append(cond_eps + wt * (cond_eps - uncond_eps))
            model_output = torch.stack(model_output, dim=0).mean(dim=0)'''
            model_output = uncond_eps.clone()
            #print(config.config['comp'])
            if config.config['comp']:
                #task_embed = [task_embed[0]]
                #print(self.wt, len(task_embed))
                for embed in task_embed:
                    cond_eps = self.model(x, x_cond, t, embed, use_dropout=False)
                    wt = 5
                    model_output += wt * (cond_eps - uncond_eps) / len(task_embed)
            else:
                cond_eps = self.model(x, x_cond, t, task_embed, use_dropout=False)
                model_output = cond_eps.clone()
                model_output += 16 * (cond_eps - uncond_eps)
        else:
            #uncond_eps = self.model(torch.cat([x_cond, x], dim=1), t, task_embed, force_dropout=True)
            model_output = self.model(x, x_cond, t, None, use_dropout=False)
        

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_cond, task_embed,  clip_denoised = False):
        preds = self.model_predictions(x, t, x_cond, task_embed)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_cond, task_embed):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x, batched_times, x_cond, task_embed, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, x_cond, task_embed, return_all_timesteps=False):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            # self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, x_cond, task_embed)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def ddim_sample(self, shape, x_cond, task_embed, return_all_timesteps=False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            # self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, x_cond, task_embed, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def sample(self, x_cond, task_embed, batch_size = 16, return_all_timesteps = False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size[0], image_size[1]), x_cond, task_embed,  return_all_timesteps = return_all_timesteps)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, x_cond, task_embed, weight, noise=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # predict and take gradient step
        #import pdb
        #pdb.set_trace()
        #x_cond = x_cond + torch.randn_like(x_cond) * 
        
        if len(task_embed) > 0:
            if not config.config['comp']:
                #print([t[0] for t in task_embed])
                model_out = self.model(x, x_cond, t, task_embed)
            else:
                model_out = []
                for embed in task_embed:
                    model_out.append(self.model(x, x_cond, t, embed))
                #print_gpu_utilization()
                model_out = torch.stack(model_out, dim=0).mean(0)
        else:
            model_out = self.model(x, x_cond, t, None)
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        
        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        loss = reduce(loss, 'b c -> b', 'mean')
        #loss = loss * weight
        if loss.mean() > 20:
            print('warning!!!', loss, weight)
        return loss.mean()

    def forward(self, img, img_cond, task_embed, weight):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}, got({h}, {w})'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, img_cond, task_embed, weight)

# dataset classes


from torch.distributed.elastic.utils.data import ElasticDistributedSampler
import imageio
from torchvision.transforms import ToPILImage
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from glob import glob
import random

import time
from collections import deque
import logging
from torchvideotransforms import video_transforms, volume_transforms

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5EncoderModel

#import deepspeed

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        tokenizer, 
        text_encoder, 
        train_set,
        valid_set,
        channels = 3,
        *,
        train_batch_size = 1,
        valid_batch_size = 1,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 3,
        results_folder = './results',
        amp = True,
        fp16 = True,
        split_batches = True,
        convert_image_to = None,
        calculate_fid = False,
        inception_block_idx = 2048, 
        cond_drop_chance=0.1,
        device=None,
        process_number=0,
        image_size = (128, 128),
        pipeline=None,
        start_multi = 40000,
    ):
        super().__init__()

        self.cond_drop_chance = cond_drop_chance

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.pipeline = pipeline

        self.gpu_device = device
        self.process_number = process_number
        self.start_multi = start_multi
        self.save_text = []
        # accelerator
        if use_accelerate:
            self.accelerator = Accelerator(
                split_batches = split_batches,
                mixed_precision = 'fp16' if fp16 else 'no'
            )

            self.accelerator.native_amp = amp
        
        self.config = config.config
        # model

        self.model = diffusion_model

        self.channels = channels

        # InceptionV3 for fid-score computation

        self.inception_v3 = None

        if calculate_fid:
            assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
            self.inception_v3 = InceptionV3([block_idx])
            self.inception_v3.to(self.gpu_device)

        # sampling and training hyperparameters

        # assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = image_size

        # dataset and dataloader

        train_set = train_set
        sampler = ElasticDistributedSampler(train_set)
        
        self.ds = train_set
        self.valid_ds = valid_set
        if self.config['ssr']:
            self.collate_fn = None
        else:
            self.collate_fn = None
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = False, \
                        pin_memory = False, num_workers = 8, sampler=sampler, collate_fn=self.collate_fn)
        if use_accelerate:
            dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        self.valid_dl = DataLoader(self.valid_ds, batch_size = valid_batch_size, shuffle = False, \
                                   pin_memory = False, num_workers = 1, collate_fn=self.collate_fn)
        self.valid_dl = cycle(self.valid_dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        #if not self.use_accelerate or self.accelerator.is_main_process:
        if self.main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0
        
        # prepare model, dataloader, optimizer with accelerator

        if use_accelerate:
            self.model, self.opt, self.text_encoder = \
                self.accelerator.prepare(self.model, self.opt, self.text_encoder)

        from diffusers import DiffusionPipeline
        pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", local_files_only=True)
        self.vae = pipeline.vae
        self.vae.requires_grad_(False)
        self.vae.to(self.device)

        self.random_state = np.random.RandomState(int(os.environ['RANK']))
        self.test_type = None

        if self.config['preload']:
            name = 'fractal20220817_data'
            with open(f'../clear_dataset/{name}/text_embeddings.pkl', 'rb') as f:
                self.text_embeddings = pickle.load(f)


    def set_dataset(self, ds):
        self.valid_ds = ds
        self.valid_dl = DataLoader(self.valid_ds, batch_size = self.valid_batch_size, shuffle = False, \
                                   pin_memory = True, num_workers = 1, collate_fn=self.collate_fn)
        self.valid_dl = cycle(self.valid_dl)

    @property
    def device(self):
        if use_accelerate:
            return self.accelerator.device
        return self.gpu_device
    
    @property
    def main_process(self):
        if use_accelerate:
            return self.accelerator.is_main_process
        return (self.process_number==0)

    def save(self, milestone):
        if use_accelerate:
            if not self.accelerator.is_local_main_process:
                return
        else:
            if not self.main_process:
                return
        
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
        if milestone != 'latest':
            torch.save(data, str(self.results_folder / f'model-latest.pt'))

    def load(self, milestone):

        device = self.device

        if not os.path.exists(str(self.results_folder / f'model-{milestone}.pt')):
            return
        path = str(self.results_folder / f'model-{milestone}.pt')
        data = None
        try:
            data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=torch.device(device))
        except:
            pt_list = glob(str(self.results_folder / f'model-*.pt'))
            milestone = len(pt_list) - 2
            while milestone >= 0:
                try:
                    data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=torch.device(device))
                    break
                except:
                    pass
                milestone -= 1
        if data is None:
            return
        if use_accelerate:
            model = self.accelerator.unwrap_model(self.model)
        self.model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])

        if self.main_process:
            self.ema.load_state_dict(data["ema"])
            print('load model step:', self.step, milestone)


    @torch.no_grad()
    def calculate_activation_statistics(self, samples):
        assert exists(self.inception_v3)
        features = self.inception_v3(samples)[0]
        features = rearrange(features, '... 1 1 -> ...')
        mu = torch.mean(features, dim = 0).cpu()
        sigma = torch.cov(features).cpu()
        return mu, sigma

    def fid_score(self, real_samples, fake_samples):

        if self.channels == 1:
            real_samples, fake_samples = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c = 3), (real_samples, fake_samples))
        real_samples = rearrange(real_samples, 'b n ... -> (b n) ...')
        fake_samples = rearrange(fake_samples, 'b n ... -> (b n) ...')
        min_batch = min(real_samples.shape[0], fake_samples.shape[0])
        real_samples, fake_samples = map(lambda t: t[:min_batch], (real_samples, fake_samples))

        m1, s1 = self.calculate_activation_statistics(real_samples)
        m2, s2 = self.calculate_activation_statistics(fake_samples)

        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value
    
    def encode_batch_text(self, batch_text):
        device = self.device
        batch_text_ids = self.tokenizer(batch_text, return_tensors = 'pt', padding = True, truncation = True, max_length = 128)
        input_ids = batch_text_ids.input_ids.to(device)
        attention_mask = batch_text_ids.attention_mask.to(device)
        if self.config['preload']:
            batch_text_embed = []
            for text in batch_text:
                text_embedding = self.text_embeddings[text]
                text_embedding = np.concatenate([text_embedding, np.zeros((128 - text_embedding.shape[0], text_embedding.shape[1]))], 0)
                batch_text_embed.append(torch.tensor(text_embedding)[:attention_mask.shape[1]].float())
            batch_text_embed = torch.stack(batch_text_embed, dim=0).to(device)
        else:
            batch_text_embed = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.detach()
        batch_text_embed = batch_text_embed.masked_fill(~attention_mask.unsqueeze(-1).bool(), 0)
        return batch_text_embed

    def image2video(self, gt_img, pred_img, path, save_idx):
        b, n, c, w, h = gt_img.shape
        to_pil = ToPILImage()
        for i in range(b):
            video_idx = save_idx * b + i
            images = []
            for j in range(n):
                img = pred_img[i, j]
                img = to_pil(img)
                img = torch.cat([gt_img[i, j], pred_img[i, j]], -1)
                img = to_pil(img)
                images.append(img)
            imageio.mimsave(f'{path}/{video_idx}.mp4', images, fps=2)
        

    def test(self, save_idx = None, name = ''):
        device = self.device
        #if accelerator.is_main_process:
        path = str(self.results_folder / f'imgs/i_{name}')
        os.makedirs(path, exist_ok = True)
        os.makedirs(str(self.results_folder / f'imgs/eval_{name}'), exist_ok=True)
        if self.main_process:
            self.ema.ema_model.eval()
            
            with torch.no_grad():
                milestone = self.step // self.save_and_sample_every
                if save_idx is None:
                    save_idx = milestone
                K = 1
                batches = num_to_groups(self.valid_batch_size * K, self.valid_batch_size)
                ### get val_imgs from self.valid_dl
                x_conds = []
                xs = []
                task_embeds = []
                all_xs_list = []

                if self.config['ssr']:
                    x, x_cond, goal, weight, goal_images = next(self.valid_dl)
                    goal_embed = None
                else:
                    x, x_cond, goal, weight, goal_images = self.get_data(self.valid_dl)
                    goal_embed = None
                    
                            
                gt_img = x.clone()
                #gt_img = (gt_img + 1) / 2
                gt_img = rearrange(gt_img, 'b (n c) h w -> b n c h w', c=3)
                xs.append(gt_img.to(device))

                x_conds.append((x_cond.clone()))
                x, x_cond = x.to(device), x_cond.to(device)
                if goal_images is not None:
                    goal_images = goal_images.to(device)
                x, x_cond, task_embed = self.preprocess(x, x_cond, goal, goal_images, goal_embed, train=False)
                print(goal)
                #self.save_text.append(' # '.join(goal))
                self.save_text.extend(goal)
                #self.ema.ema_model
                output = self.ema.ema_model.module.sample(
                    batch_size=self.valid_batch_size,
                    x_cond=x_cond,
                    task_embed=task_embed,
                )
                b = self.valid_batch_size
                if self.config['latent']:
                    output = rearrange(output, 'b (n c) h w -> (b n) c h w', c=4)
                    output = output / self.vae.config.scaling_factor
                    output = self.vae.decode(output).sample
                    output = output.clamp(-1, 1)
                    output = (output + 1) / 2
                    output = rearrange(output, '(b n) c h w -> b n c h w', b=b)
                    all_xs_list = [output]
                    '''x = rearrange(x, 'b (n c) h w -> (b n) c h w', c=4)
                    x = x / self.vae.config.scaling_factor
                    x = self.vae.decode(x).sample
                    x = rearrange(x, '(b n) c h w -> b n c h w', b=b)
                    x = (x + 1) / 2
                    xs.append(x)'''
                else:
                    output = rearrange(output, 'b (n c) h w -> b n c h w', c=3)
                    all_xs_list = [output]
                
                print('fid:', self.fid_score(xs[0], all_xs_list[0]))
                gt_xs = torch.cat(xs, dim = 0).detach().cpu() # [batch_size, n, 3, 120, 160]
                n_rows = gt_xs.shape[1]
                ### save images
                x_conds = torch.cat(x_conds, dim = 0).detach().cpu()
                gt_xs = torch.cat((x_conds[:, None], gt_xs), 1)
                
                all_xs = torch.cat(all_xs_list, dim = 0).detach().cpu()
                all_xs = torch.cat((x_conds[:, None], all_xs), 1)
                
                loss = F.mse_loss(all_xs.cpu(), gt_xs.cpu(), reduction = 'none')
                loss = reduce(loss, 'b ... -> b (...)', 'mean').mean(-1)
                print('test loss:', loss)

                os.makedirs(str(self.results_folder / f'imgs/eval'), exist_ok = True)
                n_rows += 1
                gt_img = torch.cat([gt_xs[:, :]], dim=1)
                pred_img = torch.cat([all_xs[:, :]], dim=1)
                self.image2video(gt_img, pred_img, path, save_idx)

                with open(f'{path}/{save_idx}.pkl', 'wb') as f:
                    pickle.dump([pred_img, gt_img], f)

                pred_output = []
                for i in range(pred_img.shape[0]):
                    pred_output.append(gt_img[i])
                    pred_output.append(pred_img[i])
                pred_img = torch.stack(pred_output, dim=0)
                gt_img = rearrange(gt_img, 'b n c h w -> (b n) c h w', n=n_rows)
                
                utils.save_image(gt_img, str(self.results_folder / f'imgs/eval_{name}/gt_img-{save_idx}.png'), nrow=n_rows)

                pred_img = rearrange(pred_img, 'b n c h w -> (b n) c h w', n=n_rows)
                utils.save_image(pred_img, str(self.results_folder / f'imgs/eval_{name}/sample-{save_idx}.png'), nrow=n_rows)


    def calc_time(self, num=0):
        print('time', num, time.time() - self.tmp_time)
        self.tmp_time = time.time()

    def preprocess(self, x, x_cond, goal, goal_images, goal_embed, train=True):
        device = self.device
        task_embed = []
        if self.config['ssr']:
            task_embed_list = []
            if train:
                if self.random_state.random() < 0.5:
                    x_cond += torch.randn_like(x_cond) * 0.07
        else:
            if self.config['multi']:
                task_embed_all = ['text', 'image', 'sketch']
                task_embed_list = []
                for k in task_embed_all:
                    if self.random_state.random() < 0.5:
                        continue
                    task_embed_list.append(k)
                if self.config['comp']:
                    if self.step < self.start_multi and len(task_embed_list) > 1:
                        task_embed_list = [self.random_state.choice(task_embed_list)]
                    if self.step > self.start_multi and len(task_embed_list) > 1:
                        if len(task_embed_list) == 3 or 'text' not in task_embed_list:
                            task_embed_list = ['text', self.random_state.choice(['sketch', 'image'])]
                else:
                    
                    if 'text' not in task_embed_list:
                        task_embed_list.append('text')
                    if len(task_embed_list) == 3:
                        task_embed_list = ['text', self.random_state.choice(['sketch', 'image'])]
                    if not train and self.random_state.random() < 0.2:
                        task_embed_list = []

                if not train and self.test_type is not None:
                    if self.step > self.start_multi:
                        task_embed_list = self.test_type
            else:
                task_embed_all = ['text']
                task_embed_list = []
                for k in task_embed_all:
                    if self.random_state.random() < 0.2:
                        continue
                    task_embed_list.append(k)
            if not train and len(task_embed_list) == 0:
                task_embed_list.append('text')
            if not train:
                print('task embed:', task_embed_list)
                logging.info(f'task embed: {task_embed_list}')
        # print(task_embed_list)
        if 'text' in task_embed_list:
            if goal_embed is None:
                
                goal = [g.split('#') for g in goal]
                max_len = np.max([len(g) for g in goal])
                # print('len:', max_len, goal)
                for j in range(max_len):
                    tmp_text = []
                    for i in range(len(goal)):
                        if j < len(goal[i]):
                            tmp_text.append(goal[i][j])
                        else:
                            tmp_text.append('')
                    goal_embed = self.encode_batch_text(tmp_text)
                    task_embed.append(('text', goal_embed))
            else:
                task_embed.append(('text', goal_embed))


        b = x_cond.shape[0]
        
        if self.config['latent']:
            x = x * 2 - 1
            x_cond = x_cond * 2 - 1
            x = rearrange(x, 'b (n c) h w -> (b n) c h w', c=3)
            if self.config['ssr']:
                x_cond = rearrange(x_cond, 'b (n c) h w -> (b n) c h w', c=3)
            with torch.no_grad():
                x_cond = self.vae.encode(x_cond).latent_dist.mode().detach()
                x = self.vae.encode(x).latent_dist.sample().detach()
            x_cond = x_cond * self.vae.config.scaling_factor
            x = x * self.vae.config.scaling_factor
            x = rearrange(x, '(b n) c h w -> b (n c) h w', b=b)
            if self.config['ssr']:
                x_cond = rearrange(x_cond, '(b n) c h w -> b (n c) h w', b=b)
        if 'image' in task_embed_list or 'sketch' in task_embed_list:
            goal_images = goal_images * 2 - 1
            goal_images = rearrange(goal_images, 'b c n h w->(b n) c h w')
            with torch.no_grad():
                goal_images = self.vae.encoder(goal_images).detach()
            goal_images = rearrange(goal_images, '(b n) c h w -> b n c h w', b=b)
            if 'image' in task_embed_list:
                task_embed.append(('image', goal_images[:, 1]))
            if 'sketch' in task_embed_list:
                task_embed.append(('sketch', goal_images[:, 0]))
        return x, x_cond, task_embed

    def get_data(self, dl):
        if self.config['simple']:
            x, x_cond, goal = next(dl)
            weight = None
            goal_images = None
            goal_embed = None
        else:
            x, x_cond, goal, weight, goal_images = next(dl)
        return x, x_cond, goal, weight, goal_images

    def train(self):
        device = self.device
        if True:
            total_process = int(os.environ['WORLD_SIZE'])
            self.tmp_time = time.time()
            self.start_time = time.time()
            self.start_step = self.step - 1
            if self.main_process:
                self.loss_q = deque(maxlen=200)
            while self.step < self.train_num_steps:
                
                total_loss = 0.
                for _ in range(1):
                    
                    if self.config['ssr']:
                        x, x_cond, goal, weight, goal_images = next(self.dl)
                        goal_embed = None
                    else:
                        x, x_cond, goal, weight, goal_images = self.get_data(self.dl)
                        goal_embed = None
                        
                    x, x_cond = x.to(device), x_cond.to(device)
                    if goal_images is not None:
                        goal_images = goal_images.to(device)
                    x, x_cond, task_embed = self.preprocess(x, x_cond, goal, goal_images, goal_embed)               
                    loss = self.model(x, x_cond, task_embed, None)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                if self.step % 600 == 0:
                    dist.all_reduce(loss)
                    if self.main_process:
                        print(f'step: {self.step}, reduce loss: {loss.item() / total_process}, train loss: {total_loss}')
                        logging.info(f'step: {self.step}, reduce loss: {loss.item() / total_process}, train loss: {total_loss}')
                        reduce_loss = float(loss.item() / total_process)
                        

                if self.main_process:
                    self.loss_q.append(total_loss)
                    if self.step % 50 == 0:
                        print(f'step: {self.step}, train loss: {np.mean(self.loss_q)}, max loss: {np.max(self.loss_q)}')
                        logging.info(f'step: {self.step}, train loss: {np.mean(self.loss_q)}, max loss: {np.max(self.loss_q)}')
                        print_gpu_utilization()
                
                
                self.opt.step()
                self.opt.zero_grad()
                
                if self.main_process:
                    if self.step % 100 == 0:
                        time_inter = time.time() - self.start_time
                        print('time:', self.step, time_inter, (self.step - self.start_step) / time_inter)
                        logging.info(f'time: {self.step}, {time_inter}, {(self.step - self.start_step) / time_inter}')
                    self.ema.update()
                    if self.step != 0 and self.step % (self.save_and_sample_every // 2) == 0 and total_process > 4:
                        milestone = self.step // self.save_and_sample_every
                        self.save('latest')
                    if self.step >= 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()
                        milestone = self.step // self.save_and_sample_every
                        if self.step > 0 and total_process > 4:
                            self.save(milestone)
                        with torch.no_grad():
                            batches = num_to_groups(self.valid_batch_size, self.valid_batch_size)
                            ### get val_imgs from self.valid_dl
                            x_conds = []
                            xs = []
                            task_embeds = []
                            all_xs_list = []
                            
                            if self.config['ssr']:
                                x, x_cond, goal, weight, goal_images = next(self.valid_dl)
                                goal_embed = None
                            else:
                                x, x_cond, goal, weight, goal_images = self.get_data(self.valid_dl)
                                goal_embed = None
                                
                            
                            gt_img = x.clone()
                            gt_img = rearrange(gt_img, 'b (n c) h w -> b n c h w', c=3)
                            xs.append(gt_img.to(device))
                            all_xs_list.append(gt_img.to(device))

                            x_conds.append(x_cond.clone())
                            x, x_cond = x.to(device), x_cond.to(device)
                            if goal_images is not None:
                                goal_images = goal_images.to(device)
                            x, x_cond, task_embed = self.preprocess(x, x_cond, goal, goal_images, goal_embed, train=False)
                            
                            #
                            output = self.ema.ema_model.module.sample(
                                batch_size=self.valid_batch_size,
                                x_cond=x_cond,
                                task_embed=task_embed,
                            )
                            b = output.shape[0]
                            if self.config['latent']:
                                output = rearrange(output, 'b (n c) h w -> (b n) c h w', c=4)
                                output = output / self.vae.config.scaling_factor
                                output = self.vae.decode(output).sample
                                output = output.clamp(-1, 1)
                                output = (output + 1) / 2
                                output = rearrange(output, '(b n) c h w -> b n c h w', b=b)
                                all_xs_list.append(output)
                                x = rearrange(x, 'b (n c) h w -> (b n) c h w', c=4)
                                x = x / self.vae.config.scaling_factor
                                x = self.vae.decode(x).sample
                                x = rearrange(x, '(b n) c h w -> b n c h w', b=b)
                                x = (x + 1) / 2
                                xs.append(x)
                            else:
                                output = rearrange(output, 'b (n c) h w -> b n c h w', c=3)
                                all_xs_list.append(output)

                            
                        
                        print_gpu_utilization()
                        
                        gt_xs = torch.cat(xs, dim = 0) # [batch_size, 3*n, 120, 160]
                        # make it [batchsize*n, 3, 120, 160]
                        n_rows = gt_xs.shape[1]
                        #gt_xs = rearrange(gt_xs, 'b (n c) h w -> b n c h w', n=n_rows)
                        ### save images
                        x_conds = torch.cat(x_conds, dim = 0).detach().cpu()
                        # x_conds = rearrange(x_conds, 'b (n c) h w -> b n c h w', n=1)
                        all_xs = torch.cat(all_xs_list, dim = 0).detach().cpu()
                        #all_xs = rearrange(all_xs, 'b (n c) h w -> b n c h w', n=n_rows)

                        os.makedirs(str(self.results_folder / f'imgs/outputs'), exist_ok = True)
                        gt_img = torch.cat([gt_xs[:, :]], dim=1)
                        gt_img = rearrange(gt_img, 'b n c h w -> (b n) c h w', n=n_rows)
                        utils.save_image(gt_img, str(self.results_folder / f'imgs/outputs/gt_img-{milestone}s.png'), nrow=n_rows)

                        
                        pred_img = torch.cat([all_xs[:, :]], dim=1)
                        pred_img = rearrange(pred_img, 'b n c h w -> (b n) c h w', n=n_rows)
                        utils.save_image(pred_img, str(self.results_folder / f'imgs/outputs/sample-{milestone}.png'), nrow=n_rows)
                        
                self.step += 1


    def process_img(self, img):
        h, w = img[0].size
        size = min(h, w)
        transform = video_transforms.Compose([
            video_transforms.CenterCrop((size, size)),
            video_transforms.Resize((64, 64)),
            volume_transforms.ClipToTensor()
        ])
        img = transform(img)
        return img


    
        path = '../more_zero_shot/drawers'
        N = 36
        imgs = []
        text = []
        embeds = []
        transform = None
        self.ema.ema_model.eval()
        output_path = f'../more_zero_shot/results{prefix}'
        os.makedirs(output_path, exist_ok = True)
        pretrained_model = "google/flan-t5-xxl"
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model, local_files_only=True)
        self.text_encoder = T5EncoderModel.from_pretrained(pretrained_model, local_files_only=True)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.text_encoder.to(self.device)
        with open(f'{path}/task.txt', 'r') as f:
            for line in f:
                text.append(f'Task: {line.strip()}')
            for i in range(N // 4):
                embeds.append(self.encode_batch_text(text[i*4:i*4+4]).detach())
        whole_idx = 0
        for j in range(24, N):
            i = j
            print(i)
            if os.path.exists(f'{path}/{i}.jpg'):
                img = [Image.open(f'{path}/{i}.jpg')]
            else:
                img = [Image.open(f'{path}/{i}.webp')]
            img = self.process_img(img)[:, 0]
            print(img.shape)
            imgs.append(img)
            if len(imgs) == 4:
                x_cond = torch.stack(imgs, 0).to(self.device)
                #task_embed = self.encode_batch_text(text[i-3: i+1])
                task_embed = [('text', embeds[i//4])]
                images = self.ema.ema_model.module.sample(batch_size=4, x_cond=x_cond, task_embed=task_embed).detach().cpu()
                x_cond = x_cond.cpu()
                c=3
                images = rearrange(images, 'b (n c) h w->b n c h w', c=3)
                images = torch.cat([x_cond[:, None], images], 1)

                #import pdb; pdb.set_trace()
                for idx in range(images.shape[0]):
                    imgs = images[idx]
                    img_list = []
                    for t in range(imgs.shape[0]):
                        image = imgs[t].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                        #image = to_pil(images[i])
                        #image.save(f'wild/{i}.jpg')
                        image = Image.fromarray(image)
                        img_list.append(image)
                    img_list.append(image)
                    img_list.append(image)
                    #img_list[-1].save(f'{output_path}/{whole_idx}_final{prefix}.jpg')
                    imageio.mimsave(f'{output_path}/{j+idx}.mp4', img_list, fps=2)
                    whole_idx += 1

                images = rearrange(images, 'b n c h w->(b n) c h w')
                utils.save_image(images, f'{output_path}/{j//4}.png', nrow=8)
                imgs = []