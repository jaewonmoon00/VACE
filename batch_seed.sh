#!/bin/bash

# 시작 시드와 종료 시드 설정
START_SEED=2026
END_SEED=2032

# 비디오 경로
VIDEO_PATH="/data/VACE/inputs/fantastic4_2007-Scene-183_chunk1.mp4"

# 각 시드에 대해 명령 실행
for seed in $(seq $START_SEED $END_SEED); do
    echo "Running with base_seed: $seed"
    
    CUDA_VISIBLE_DEVICES=0 python vace/vace_pipeline.py \
        --base wan \
        --task outpainting \
        --direction 'left,right' \
        --expand_ratio 1.6 \
        --video "$VIDEO_PATH" \
        --prompt "" \
        --base_seed $seed
    
    echo "Completed seed: $seed"
    echo "-----------------------"
done

echo "모든 시드 처리 완료"
