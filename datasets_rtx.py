from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
import torch
from utils import get_paths, get_paths_from_dir
from tqdm import tqdm
from PIL import Image
import numpy as np
import json
import torchvision.transforms as T
import random
from torchvideotransforms import video_transforms, volume_transforms
from einops import rearrange
import cv2
from transformers import T5Tokenizer, T5EncoderModel
import pandas as pd
import time
import json
import copy
import imageio

from config import config, init_config

random.seed(0)

import pickle

from utils import *

from einops import rearrange
import re


dataset_min_length = {
    'fractal20220817_data': 20,
}

dataset_max_length = {
    'fractal20220817_data': 100,
}

dataset_skip_frame = {
    'fractal20220817_data': 5,
}

dataset_weight = {
    'fractal20220817_data': 0.5,
}

def decode_inst(inst):
    return bytes(inst[np.where(inst != 0)].tolist()).decode('utf-8')



import pickle

class XDataset(Dataset):

    def __init__(self, path=None, name="", sample_per_seq=7, target_size=(128, 128), frame_skip=None, randomcrop=False, train=True, pre=False, verbose=False, seed=None):
        #print("preparing dataset...")
        min_length = dataset_min_length.get(name, 0)
        max_length = dataset_max_length.get(name, 1000)
        self.sample_per_seq = sample_per_seq
        frame_skip = dataset_skip_frame.get(name, frame_skip)
        self.frame_skip = frame_skip
        self.name = name
        self.path = path
        self.train = train
        self.tasks = []
        self.sequences = []
        self.narration_ids = []

        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        
        path = f'{path}/{name}'
        self.path = path
        # analysis the dataset
        self.num_frames = []
        try:
            with open(f'{path}/meta_info_1.pkl', 'rb') as f:
                self.meta_info = pickle.load(f)
        except:
            with open(f'{path}/meta_info_0.pkl', 'rb') as f:
                self.meta_info = pickle.load(f)
        N = self.meta_info['number']
        self.length = np.array(self.meta_info['length'])
        
        if verbose:
            # filter values
            self.length = self.length[self.length >= min_length]
            self.length = self.length[self.length <= max_length]

            print(f'dataset {name}, total {len(self.length)} sequences, mean length {self.length.mean()}, min length {self.length.min()}, max length {self.length.max()}')

        self.ids = list(range(N))
        # filter based on self.length
        self.ids = [i for i in self.ids if self.meta_info['length'][i] >= min_length and self.meta_info['length'][i] <= max_length]

        if verbose:
            print('total after filter', len(self.ids))
        random.Random(9).shuffle(self.ids)
        N = len(self.ids)
        M = int(N * 0.9)
        if self.train:            
            
            with open(f'data_pkl/{name}_train.pkl', 'rb') as f:
                self.ids = pickle.load(f)
            random.Random(seed).shuffle(self.ids)
        else:
            with open(f'data_pkl/{name}_test.pkl', 'rb') as f:
                self.ids = pickle.load(f)
        sample = self.get_samples(0)[0]
        H, W, _ = np.array(Image.open(sample)).shape
        size = min(H, W)
        #print('size:', H, W)
        self.transform = video_transforms.Compose([
            video_transforms.CenterCrop((size, size)),
            video_transforms.Resize(target_size),
            volume_transforms.ClipToTensor()
        ])
        self.crop_transform = video_transforms.Compose([
            video_transforms.RandomCrop((size, size)),
            video_transforms.Resize(target_size),
            volume_transforms.ClipToTensor()
        ])

        with open(f'{path}/parse_text.pkl', 'rb') as f:
            self.parse_result = pickle.load(f)
        

    def __len__(self):
        return len(self.ids)

    def get_samples(self, idx, start_idx = None):
        # seq = self.sequences[idx]
        # if frameskip is not given, do uniform sampling betweeen a random frame and the last frame
        max_len = self.meta_info['length'][idx]
        if self.frame_skip is None:
            start_idx = self.random_state.randint(0, max_len-1)
            N = max_len - start_idx
            samples = []
            for i in range(self.sample_per_seq-1):
                samples.append(start_idx + int(i*(N-1)/(self.sample_per_seq-1)))
            samples.append(start_idx + N-1)
        else:
            if self.train:
                start_idx = np.random.randint(0, max(max_len-self.frame_skip * (self.sample_per_seq-1) // 3, max_len // 2))
            else:
                start_idx = np.random.randint(0, max(max_len-self.frame_skip * (self.sample_per_seq-1) // 3, max_len // 4))
            if not self.train:
                start_idx = 0
            samples = [i if i < max_len else max_len-1 for i in range(start_idx, start_idx+self.frame_skip*self.sample_per_seq, self.frame_skip)]
        return [f'{self.path}/ep{idx:06d}/step{i:04d}.jpg' for i in samples]

    def __getitem__(self, idx):
        try:
            idx = self.ids[idx]
            
            samples = self.get_samples(idx)
            images = [Image.open(s) for s in samples]
            soft = np.load(f'{self.path}/ep{idx:06d}/soft.npy')[-1]
            if len(soft.shape) == 3:
                soft = soft[0]
            images.append(Image.fromarray(soft))
            step_idx = self.meta_info['length'][idx]-1
            goal_path = f'{self.path}/ep{idx:06d}/step{step_idx:04d}.jpg'
            images.append(Image.open(goal_path))

            if self.random_state.random() < 0.7 and self.train:
                images = self.transform(images) # [c f h w]
            else:
                images = self.crop_transform(images) # [channel frame h w]
            x_cond = images[:, 0] # first frame
            goal_images = images[:, -2:]
            x = rearrange(images[:, 1:-2], "c f h w -> (f c) h w") # all other frames
            task = self.meta_info['language'][idx]
            if isinstance(task, bytes):
                task = task.decode('utf-8')

            if self.train:
                if config.config['parse_data']:
                    parse_tmp = self.parse_result[idx]
                else:
                    parse_tmp = []
                if len(parse_tmp) > 0:
                    rand_v = random.random()
                    if rand_v < 0.3:
                        task = [self.random_state.choice(parse_tmp)]
                    elif rand_v < 0.6:
                        task = parse_tmp
                    else:
                        task = [task]
                else:
                    task = [task]
            else:
                if config.config['parse_data']:
                    parse_tmp = self.parse_result[idx]
                else:
                    parse_tmp = []
                task = [task] + parse_tmp
            task = '#'.join(task)
            return x, x_cond, task, dataset_weight[self.name], goal_images
        except Exception as e:
            print(e, idx)       
            #raise Exception     
            return self.__getitem__(idx + 1 % self.__len__())


def set_XDataset(name, train=False):
    return XDataset(
        name=name,
        sample_per_seq=8, 
        target_size=(64, 64),
        frame_skip=dataset_skip_frame[name],
        randomcrop=False,
        train=train
    )

from torchvision.transforms import ToPILImage
to_pil = ToPILImage()
import time
