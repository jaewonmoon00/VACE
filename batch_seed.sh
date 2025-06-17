#!/bin/bash

# 실행 횟수 설정 (원하는 만큼 수정)
RUN_COUNT=1

# 비디오 경로
VIDEO_PATH="/data/VACE/inputs/272.mp4"

echo "랜덤 시드로 $RUN_COUNT 번 실행 시작"
echo "처리할 영상: $(basename "$VIDEO_PATH")"
echo "=========================="

# 처리된 횟수 카운터
processed_count=0

# 지정된 횟수만큼 반복 실행
for i in $(seq 1 $RUN_COUNT); do
    # 랜덤 시드 생성 (현재 시간 + 랜덤값)
    current_seed=$RANDOM
    
    echo "실행 #$i - 랜덤 시드: $current_seed"
    
    torchrun --nproc-per-node=2 vace/vace_pipeline.py \
        --base wan \
        --task outpainting \
        --direction 'left,right' \
        --expand_ratio 1.6 \
        --video "$VIDEO_PATH" \
        --prompt "" \
	--neg_prompt "" \
        --base_seed $current_seed \
        --dit_fsdp \
        --t5_fsdp \
        --ulysses_size 2 \
        --ring_size 1
    
    # 명령 실행 결과 확인
    if [ $? -eq 0 ]; then
        echo "완료: 실행 #$i (시드: $current_seed)"
        processed_count=$((processed_count + 1))
    else
        echo "오류 발생: 실행 #$i"
    fi
    
    echo "-----------------------"
done

echo "모든 실행 완료!"
echo "총 성공한 실행 수: $processed_count / $RUN_COUNT"
