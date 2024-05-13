#!/bin/bash
#SBATCH --nodes=24
#SBATCH --ntasks-per-node=1
#SBATCH -o logs_raw/%j.out
#SBATCH -e logs_raw/%j.err
#SBATCH --gres=gpu:32g:6
#SBATCH --mem=200GB
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --job-name=raw
#SBATCH --mail-user=elegyhunter@gmail.com

# activate conda environment
source ~/.bashrc_bk
eval "$(conda shell.bash hook)"
conda activate ego
#source scripts/cache.sh

# reference: https://huggingface.co/docs/transformers/main/main_classes/deepspeed#launching-in-a-slurm-environment
export GPUS_PER_NODE=6
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=8000
export WANDB_MODE=disabled

echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
export NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DETAIL=DEBUG

# ====
# Note: sbatch doesn't read command line arguments as expected.
# Use scripts/submit_training_jobs.py to submit this script and set {extra} there
# ====
# --deepspeed ds_config.json \

srun bash -c 'torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank $SLURM_PROCID \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    train_rtx.py \
    --batch_size 2 \
    --lr 7e-5 \
    --save_id 1 \
    --image_size 64 \
    --H 8 \
    --job_name raw'
