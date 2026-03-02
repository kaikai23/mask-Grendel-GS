#!/bin/bash
#SBATCH -N 8
#SBATCH --gres=gpu:4
#SBATCH -p vip_gpu_ailab_low
#SBATCH -A ailab
#SBATCH --qos=gpugpu
#SBATCH -t 8:00:00
#SBATCH -o myjob.04.out
#SBATCH -e myjob.04.err

echo "start"
source /home/bingxing2/ailab/liuyifei/.bashrc
conda activate grendel

# srun torchrun --standalone --nnodes=1 --nproc-per-node=4 \
#     render.py --bsz 1 -m output/rubble_pixsfm_g32_bz8_200k_50k_lrinit0.000016_0.00003_0.0008 \
#     --skip_train

export NCCL_ALGO=Ring
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_DEBUG=INFO
export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml
export NCCL_IB_HCA=mlx5_0,mlx5_2
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500
echo $MASTER_ADDR
echo $MASTER_PORT

srun torchrun --nnodes=8 --nproc-per-node=4 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    render.py --bsz 1 -m output/rubble_pixsfm_g32_bz8_200k_50k_lrinit0.000016_0.00003_0.0008 \
    --skip_train

echo "finished rubble"



