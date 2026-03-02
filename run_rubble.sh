torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py --bsz 4 -s data/rubble-pixsfm-from-fred --eval \
    --log_interval 250 \
    --test_iterations 20000 40000 60000 80000 100000 120000 140000 160000 180000 200000 \
    --save_iterations 200000 \
    --preload_dataset_to_gpu_threshold 0 \
    --lambda_mask 0.0005 \
    --mask_from_iter 0 \
    --mask_until_iter 200000 \
    -m output/rubble_pixsfm_m0.0005_200k_50k_0.00013_0.003 \
    --iterations 200000 \
    --densify_until_iter 50000 \
    --densify_grad_threshold 0.00013 \
    --percent_dense 0.003

echo "finished rubble"



