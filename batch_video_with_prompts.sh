#!/bin/bash

# 영상이 있는 폴더 경로 설정
INPUT_FOLDER="/data/VACE/inputs/"
# 파일-프롬프트 쌍이 저장된 파일 (파이프 구분자 사용)
PROMPT_FILE="/data/VACE/video_prompts.txt"

echo "폴더 내 영상 처리 시작: $INPUT_FOLDER"
echo "프롬프트 파일: $PROMPT_FILE"
echo "=========================="

# 처리된 파일 카운터
processed_count=0

# 프롬프트 파일이 존재하는지 확인
if [ ! -f "$PROMPT_FILE" ]; then
    echo "오류: 프롬프트 파일을 찾을 수 없습니다: $PROMPT_FILE"
    exit 1
fi

# 프롬프트 파일을 읽어서 처리 (파이프 구분자 사용)
while IFS='|' read -r filename prompt; do
    # 빈 줄이나 주석 건너뛰기
    if [[ -z "$filename" || "$filename" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    # 앞뒤 공백 제거
    filename=$(echo "$filename" | xargs)
    prompt=$(echo "$prompt" | xargs)
    
    # 파일 경로 구성
    video_file="$INPUT_FOLDER$filename"
    
    # 파일이 실제로 존재하는지 확인 (인코딩 문제 해결)
    if [ -f "$video_file" ]; then
        echo "처리 중: $filename"
        echo "프롬프트: $prompt"
        
        current_seed=$RANDOM
        # VACE 파이프라인 실행
        torchrun --nproc-per-node=2 vace/vace_pipeline.py --base wan --task outpainting --direction 'left,right' --expand_ratio 1.6 --video "$video_file" --prompt "$prompt" --base_seed $current_seed --dit_fsdp --t5_fsdp --ulysses_size 2 --ring_size 1
        
        if [ $? -eq 0 ]; then
            echo "완료: $filename (시드: $current_seed)"
            processed_count=$((processed_count + 1))
        else
            echo "오류 발생: $filename"
        fi
        echo "-----------------------"
    else
        echo "파일을 찾을 수 없습니다: $video_file"
    fi
done < "$PROMPT_FILE"

if [ $processed_count -eq 0 ]; then
    echo "처리할 영상 파일을 찾을 수 없습니다."
    echo "프롬프트 파일과 경로를 확인해주세요."
else
    echo "모든 영상 처리 완료!"
    echo "총 처리된 파일 수: $processed_count"
fi
