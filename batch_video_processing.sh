#!/bin/bash

# 영상이 있는 폴더 경로 설정
INPUT_FOLDER="/data/VACE/inputs/"

# 사용할 시드값 (모든 영상에 동일하게 적용하거나, 파일마다 다르게 하려면 수정)
BASE_SEED=2025

# 지원하는 영상 확장자 패턴
VIDEO_EXTENSIONS="*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"

echo "폴더 내 영상 처리 시작: $INPUT_FOLDER"
echo "사용할 시드: $BASE_SEED"
echo "=========================="

# 처리된 파일 카운터
processed_count=0

# 폴더 내 모든 영상 파일을 찾아서 처리
for ext in $VIDEO_EXTENSIONS; do
    for video_file in "$INPUT_FOLDER"$ext; do
        # 파일이 실제로 존재하는지 확인
        if [ -f "$video_file" ]; then
            filename=$(basename "$video_file")
            echo "처리 중: $filename"
            
            # 파일마다 다른 시드를 사용하려면 아래 라인의 주석을 해제하고 위의 BASE_SEED 사용 부분을 주석처리
            # current_seed=$((BASE_SEED + processed_count))
	    current_seed=$(($(date + %s) + $RANDOM))
            
	    echo "사용할 랜덤 시드: $current_seed"

            torchrun --nproc-per-node=2 vace/vace_pipeline.py \
                --base wan \
                --task outpainting \
                --direction 'left,right' \
                --expand_ratio 1.6 \
                --video "$video_file" \
                --prompt "" \
                --base_seed $current_seed \
		--dit_fsdp \
		--t5_fsdp \
		--ulysses_size 2 \
		--ring_size 1
            
            if [ $? -eq 0 ]; then
                echo "완료: $filename (시드: $current_seed)"
                processed_count=$((processed_count + 1))
            else
                echo "오류 발생: $filename"
            fi
            echo "-----------------------"
        fi
    done
done

if [ $processed_count -eq 0 ]; then
    echo "처리할 영상 파일을 찾을 수 없습니다."
    echo "폴더 경로를 확인해주세요: $INPUT_FOLDER"
    echo "지원 확장자: $VIDEO_EXTENSIONS"
else
    echo "모든 영상 처리 완료!"
    echo "총 처리된 파일 수: $processed_count"
fi
