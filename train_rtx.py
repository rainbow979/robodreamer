from goal_diffusion_rtx import GoalGaussianDiffusion, Trainer, print_gpu_utilization
from unet import Unet
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTokenizerFast
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5EncoderModel

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import argparse
import os
os.environ['CURL_CA_BUNDLE'] = ''
import datetime

import logging

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--save_id', default=0)
    parser.add_argument('--H', type=int, default=8)
    parser.add_argument('--skip', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--job_name', default='rtx_test')
    parser.add_argument('--name', default='fractal20220817_data')
    args = parser.parse_args()

    #DDP
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    total_process = int(os.environ["WORLD_SIZE"])

    from config import config, init_config
    init_config(int(args.save_id))
    if not config.config['preload']:
        if config.config['text'] == 'clip':
            pretrained_model = "openai/clip-vit-base-patch32"
            tokenizer = CLIPTokenizer.from_pretrained(pretrained_model, local_files_only=True)
            text_encoder = CLIPTextModel.from_pretrained(pretrained_model, local_files_only=True)
            text_encoder.requires_grad_(False)
            text_encoder.eval()
            text_encoder.cuda(local_rank)
        else:
            pretrained_model = "google/flan-t5-xxl"
            tokenizer = T5Tokenizer.from_pretrained(pretrained_model, local_files_only=True)
            text_encoder = T5EncoderModel.from_pretrained(pretrained_model, local_files_only=True)
            text_encoder.requires_grad_(False)
            text_encoder.eval()
            text_encoder.cuda(local_rank)
    else:
        text_encoder = None
        pretrained_model = "google/flan-t5-xxl"
        tokenizer = T5Tokenizer.from_pretrained(pretrained_model, local_files_only=True)

    pipeline = None

    from datasets_rtx import XDataset

    valid_n = 30
    sample_per_seq = args.H
    frame_skip = args.skip
    if config.config['latent']:
        args.image_size = 256
    target_size = (args.image_size, args.image_size)

    name = args.name
    train_set = XDataset(
        name=name,
        sample_per_seq=sample_per_seq, 
        target_size=target_size,
        frame_skip=frame_skip,
        randomcrop=True,
        train=True,
        seed=rank,
    )
    valid_set = XDataset(
        name=name,
        sample_per_seq=sample_per_seq, 
        target_size=target_size,
        frame_skip=frame_skip,
        randomcrop=True,
        train=False,
        seed=rank,
    )

    Unet_model = Unet(image_size=args.image_size)
    if config.config['latent']:
        base_channel = 4
        image_scale = 8
        timesteps = 1000
        sampling_time = 100
    else:
        base_channel = 3
        image_scale = 1
        timesteps = 200
        sampling_time = 100
    diffusion = GoalGaussianDiffusion(
        channels=base_channel*(sample_per_seq-1),
        model=Unet_model,
        image_size=(args.image_size // image_scale, args.image_size // image_scale),
        timesteps=timesteps,
        sampling_timesteps=sampling_time,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )

    if rank == 0:
        print(config)
        total_params = 0
        for param in Unet_model.parameters():
            total_params += param.numel()
        print('total parameters:', total_params)

    args.total_process = total_process

    diffusion.cuda(local_rank)
    

    diffusion = DDP(diffusion, device_ids=[local_rank], find_unused_parameters=True)
    
    batch_size = args.batch_size
    if total_process < 3:
        if not config.config['latent']:
            save_and_sample = 200000 // batch_size // total_process
        else:
            save_and_sample = 1000000 // batch_size // total_process
    else:
        save_and_sample = 5000
    if rank == 0:
        print('save and sample:', save_and_sample)
    trainer = Trainer(
        diffusion_model=diffusion,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=train_set,
        valid_set=valid_set,
        train_lr=args.lr,
        train_num_steps =1800000,
        save_and_sample_every =save_and_sample,   #2500,
        ema_update_every = 10,
        ema_decay = 0.995,
        train_batch_size =batch_size,
        valid_batch_size =1,
        gradient_accumulate_every = 1,
        num_samples=valid_n, 
        results_folder =f'../results_ego{args.save_id}',
        fp16 =True,
        amp=True,
        device=local_rank,
        process_number=rank,
        image_size=target_size,
        pipeline=pipeline,
        start_multi=config.config['start_multi'],
    )
    trainer.load('latest')
    if rank == 0:
        if total_process > 10:
            print('logging')
            os.makedirs(f'../results_ego{args.save_id}/logs', exist_ok=True)
            logging.basicConfig(filename=f'../results_ego{args.save_id}/logs/{trainer.step}.txt', filemode='w', level=logging.INFO)
            logging.info('start training')
        else:
            logging.basicConfig(level=logging.ERROR)
    trainer.train()

if __name__ == '__main__':
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
    train()
    dist.destroy_process_group()
