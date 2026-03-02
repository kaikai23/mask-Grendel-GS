#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -p vip_gpu_ailab_low
#SBATCH -A ailab
#SBATCH --qos=gpugpu
#SBATCH -t 24:00:00
#SBATCH -o myjob.01.out
#SBATCH -e myjob.01.err

echo "start"
source /home/bingxing2/ailab/liuyifei/.bashrc
conda activate mask_grendel

srun torchrun --standalone --nnodes=1 --nproc-per-node=4 \
    render.py --bsz 1 -m output/rubble_pixsfm_g16_bz8_storedm0.0005_200k_50k_lrinit0.000016_0.0001_0.002 \
    --skip_train

echo "finished rubble"



