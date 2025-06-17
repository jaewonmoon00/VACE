# VACE Custom Video Outpainting Pipeline

본 프로젝트는 오픈소스 [VACE (All-in-One Video Creation and Editing)](https://github.com/ali-vilab/VACE) 모델을 기반으로 한 커스터마이징된 영상 아웃페인팅 파이프라인입니다. **VACE 14B 모델**을 사용하여 **1.6:1:1.6 비율로 좌우 영상 확장**을 수행하며, 다중 GPU 환경에서의 배치 처리를 지원합니다.

## 🌟 주요 기능

- **VACE 14B 모델 기반 고품질 영상 아웃페인팅**: 1.6:1:1.6 비율로 좌우 영상 확장
- **81프레임 단위 자동 분할 및 처리**: 메모리 부족 문제 해결
- **배치 처리 시스템**: 다수의 영상에 대한 자동화된 일괄 처리  
- **다중 GPU 병렬처리**: 메모리 부족 문제 해결을 위한 분산 처리
- **프롬프트 기반 영상 생성**: 씬별 맞춤형 프롬프트 적용
- **결과물 일관성 유지**: 영상 분할 시에도 자연스러운 연결

## 📋 주요 제약사항 및 해결책

### 알려진 제약사항
1. **메모리 제한**: 3초 이상 영상 처리 시 메모리 부족 문제 ([관련 이슈](https://github.com/ali-vilab/VACE/issues/56))
2. **프레임 정확성**: 입력 영상이 정확히 81프레임이 아닐 경우 결과물 프레임 수 불일치 가능
3. **한글 파일명**: 한글 파일명 인식 불가 (영문 변환 필요)

### 해결책
- **영상 분할**: `video_splitter.py`를 통한 81프레임 단위 분할 및 패딩
- **다중 GPU**: 분산 처리를 통한 메모리 부족 문제 해결
- **파일명 정규화**: 영문 파일명 자동 변환 스크립트 제공

## 🚀 시작하기

### 전제 조건

- Python 3.8+
- CUDA 지원 GPU (다중 GPU 권장, 최소 24GB VRAM)
- FFmpeg (영상 분할용)

### 설치

```bash
# VACE repository 클론
git clone https://github.com/your-username/vace.git
cd vace

# 의존성 설치
pip install -r requirements.txt
pip install wan@git+https://github.com/Wan-Video/Wan2.1

# VACE 14B 모델 다운로드
# models/ 디렉토리에 Wan2.1-VACE-14B 모델 배치
```

## 📁 디렉토리 구조

```
├── inputs/                    # 처리할 영상 파일들 (81프레임 청크)
├── outputs/                   # 결과물 저장 디렉토리  
├── pre_processed_files/       # 전처리 대상 원본 영상들
├── results/                   # 최종 결과물 저장 디렉토리
├── video_splitter.py         # 영상 분할 도구
├── batch_video_with_prompts.sh  # 배치 처리 스크립트
├── video_prompts.txt         # 프롬프트 설정 파일
└── vace_wan_inference.py     # 메인 추론 스크립트
```

## 📋 사용 방법

### 1. 영상 준비 및 분할

#### 영상을 81프레임 단위로 분할
```bash
# 단일 영상 분할
python video_splitter.py input_video.mov -o ./inputs/

# 여러 영상 일괄 분할  
for video in ./pre_processed_files/*.mov; do 
    python video_splitter.py "$video" -o ./inputs/
done
```

#### 한글 파일명 영문 변환
```bash
# 파일명 정규화 (한글 → 영문)
for file in *.mov; do
    number=$(echo "$file" | grep -o '[0-9]\{8\}' | head -1)
    chunk=$(echo "$file" | grep -o 'chunk[0-9]')
    new_name="video_${number}_${chunk}.mov"
    mv "$file" "$new_name"
done
```

### 2. 프롬프트 설정

`video_prompts.txt` 파일에 다음 형식으로 작성:

```
video_00000024_chunk1.mov|dark room, no lights, no people
video_00000057_chunk1.mov|The person in the middle is giving a speech alone in a dark room
video_00000184_chunk1.mov|A cloudless sky
video_00000227_chunk1.mov|People waving blue flags and cheering
video_00000370_chunk1.mov|
```

### 3. 영상 처리 실행

#### 단일 영상 처리
```bash
python vace/vace_pipeline.py \
    --base wan \
    --task outpainting \
    --direction 'left,right' \
    --expand_ratio 1.6 \
    --video input_video.mov \
    --prompt "your prompt here" \
    --base_seed 2025
```

#### 배치 처리 (권장)
```bash
# 프롬프트 파일 기반 배치 처리
bash batch_video_with_prompts.sh
```

#### 다중 GPU 환경에서 실행
```bash
# 2개 GPU 사용 예시
torchrun --nproc-per-node=2 vace/vace_pipeline.py \
    --base wan \
    --task outpainting \
    --direction 'left,right' \
    --expand_ratio 1.6 \
    --video "$video_file" \
    --prompt "$prompt" \
    --base_seed $current_seed \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 2 \
    --ring_size 1
```

## ⚙️ 주요 매개변수

### 필수 매개변수
- `--video`: 입력 영상 경로
- `--prompt`: 생성할 장면에 대한 텍스트 설명
- `--direction`: 확장 방향 ('left,right' 권장)
- `--expand_ratio`: 확장 비율 (1.6 권장)

### 선택적 매개변수  
- `--base_seed`: 랜덤 시드 (재현성을 위해 고정값 권장)
- `--dit_fsdp`: DiT 모델 분산 처리 활성화
- `--t5_fsdp`: T5 텍스트 인코더 분산 처리 활성화
- `--ulysses_size`: Ulysses 병렬 처리 크기
- `--ring_size`: Ring attention 병렬 처리 크기

## 🔧 고급 기능

### 모델 캐싱
모델 로딩 시간을 단축하기 위해 메모리 기반 캐싱을 지원합니다:
```bash
export MODEL_CACHE_TIMEOUT=3600  # 1시간 캐시 유지
```

### 프롬프트 자동 확장
LLM을 사용한 프롬프트 자동 확장 기능:
```bash
python vace/vace_wan_inference.py \
    --use_prompt_extend wan_zh \  # 중국어 확장
    # 또는 wan_en (영어 확장)
    --prompt "간단한 프롬프트"
```

### 커스텀 네거티브 프롬프트
```bash
python vace/vace_wan_inference.py \
    --neg_prompt "unwanted elements, blurry, distorted" \
    --prompt "your prompt"
```

## 📊 성능 최적화

### GPU 메모리 사용량
- **14B 모델**: 최소 24GB VRAM (단일 GPU)
- **분산 처리**: 2개 이상 GPU 권장 (각 12GB+)
- **배치 처리**: CPU 메모리 8GB+ 권장

### 처리 속도
- **81프레임 (3.375초)**: 약 2-5분 (GPU 성능에 따라)
- **배치 처리**: 병렬 처리로 전체 처리 시간 단축

## 🛠️ 문제 해결

### 메모리 부족
```bash
# 다중 GPU 환경 설정
torchrun --nproc-per-node=4 vace/vace_pipeline.py \
    --dit_fsdp --t5_fsdp \
    --ulysses_size 4 --ring_size 1
```

### 프레임 불일치
```bash
# 영상을 정확히 81프레임으로 분할 및 패딩
python video_splitter.py input_video.mov -o ./inputs/
```

### 한글 파일명 문제
```bash
# 파일명을 영문으로 변환 후 처리
for file in *.mov; do
    # ... 파일명 변환 스크립트
done
```

## 📋 TODO 및 개발 계획

### 높은 우선순위
- [ ] **영상 캡셔닝 도구**: 자동 프롬프트 생성
- [ ] **신별 영상 분할 개선**: 현재 불완전한 분할 로직 보완
- [ ] **UI 통합**: 전처리와 추론 UI 합병

### 중간 우선순위  
- [ ] **Video Interpolation**: 81프레임 제약 해결
- [ ] **모델 최적화**: 메모리 사용량 최적화
- [ ] **결과물 품질 개선**: 일관성 유지 알고리즘

## 📖 추가 정보

### 원본 VACE 프로젝트
자세한 VACE 모델 정보 및 다른 기능들은 [원본 VACE 프로젝트](https://github.com/ali-vilab/VACE)를 참조하세요.

### 라이선스
본 프로젝트는 Apache License 2.0 하에 배포됩니다.

### 기여
버그 리포트, 기능 제안, 풀 리퀘스트를 환영합니다.

## 🚨 주의사항

1. **GPU 메모리**: 24GB 이상의 VRAM을 권장합니다
2. **영상 길이**: 3초 이하의 영상으로 분할하여 처리하세요
3. **파일명**: 영문 파일명을 사용하세요
4. **분산 처리**: 메모리 부족 시 다중 GPU 환경을 구성하세요

---

문의사항이나 이슈가 있으시면 GitHub Issues를 통해 제보해 주세요.