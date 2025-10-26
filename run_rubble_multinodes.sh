#!/bin/bash
#SBATCH -N 4
#SBATCH --gres=gpu:4
#SBATCH -p vip_gpu_ailab_low
#SBATCH -A ailab
#SBATCH --qos=gpugpu
#SBATCH -t 10:00:00
#SBATCH -o myjob.03.out
#SBATCH -e myjob.03.err

export NCCL_ALGO=Ring
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_DEBUG=INFO
export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml
export NCCL_IB_HCA=mlx5_0,mlx5_2
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7


# NPROC_PER_NODE=4
# NODE_RANK=

echo "start"
source /home/bingxing2/ailab/liuyifei/.bashrc
conda activate grendel

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500
echo $MASTER_ADDR
echo $MASTER_PORT

srun torchrun --nnodes=4 --nproc-per-node=4 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py --bsz 8 -s data/rubble-pixsfm-from-fred --eval \
    --log_interval 250 \
    --test_iterations 20000 40000 60000 80000 100000 120000 140000 160000 180000 200000\
    --save_iterations 200000 \
    --preload_dataset_to_gpu_threshold 0 \
    -m output/rubble_pixsfm_g16_bz8_200k_50k_lrinit0.000016_0.00018_0.008 \
    --iterations 200000 \
    --densify_until_iter 50000 \
    --position_lr_init 0.000016 \
    --densify_grad_threshold 0.00018 \
    --percent_dense 0.008

echo "finished training"

srun torchrun --standalone --nnodes=1 --nproc-per-node=4 \
    render.py --bsz 1 -m output/rubble_pixsfm_g16_bz8_200k_50k_lrinit0.000016_0.00018_0.008 \
    --skip_train

echo "finished rendering"



