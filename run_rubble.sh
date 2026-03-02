torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py --bsz 4 -s data/rubble-pixsfm-from-fred --eval \
    --log_interval 250 \
    --test_iterations 20000 40000 60000 80000 100000 120000 140000 160000 180000 200000\
    --save_iterations 200000 \
    --preload_dataset_to_gpu_threshold 0 \
    -m output/rubble_pixsfm_200k_50k_0.00008_0.0016 \
    --iterations 200000 \
    --densify_until_iter 50000 \
    --densify_grad_threshold 0.00008 \
    --percent_dense 0.0016

echo "finished rubble"



