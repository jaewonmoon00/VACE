# VACE Custom Video Processing Pipeline

본 프로젝트는 오픈소스 [VACE](https://github.com/ali-vilab/VACE) 모델을 기반으로 한 커스터마이징된 영상 처리 파이프라인입니다. 영상 아웃페인팅, 자동화된 워크플로우, 그리고 최적화된 처리 과정을 통해 고품질의 영상 확장 결과물을 생성합니다.

## 🌟 주요 기능

- **VACE 14B 모델 기반 영상 아웃페인팅**: 1.6:1:1.6 비율로 좌우 영상 확장
- **배치 처리 시스템**: 다수의 영상에 대한 자동화된 일괄 처리
- **다중 GPU 병렬처리**: 메모리 부족 문제 해결을 위한 분산 처리
- **영상 분할 및 처리**: 81프레임 단위 자동 분할 및 일관성 유지
- **배치 처리**: 다수의 영상에 대한 일괄 처리 지원

## 🚀 시작하기

### 전제 조건

- Python 3.8+
- CUDA 지원 GPU (다중 GPU 권장)
- 충분한 VRAM (3초 영상 처리 시 상당한 메모리 필요)

### 설치

```bash
# 원본 VACE repository fork 후 클론
git clone https://github.com/your-username/vace.git
cd vace

# upstream 설정
git remote add upstream https://github.com/ali-vilab/VACE.git

# 의존성 설치
pip install -r requirements.txt
```

## 📋 워크플로우

### 1. 최적화 프로세스 (TA)

영상 품질 최적화를 위한 단계별 처리:

1. **초기 처리**: 최소 크기로 시작
2. **중간 리프레임**: 6번째 점까지 줄여서 리프레임
3. **최종 업스케일**: 고해상도로 업스케일
4. **후처리**: 
   - Topaz를 통한 추가 업스케일
   - 소프트웨어를 이용한 해상도 리사이즈
   - 신별 영상 분할 (페이드 인/아웃 제외)

### 2. VACE 14B 모델 아웃페인팅

#### 기본 사용법

```bash
# 단일 영상 처리
python inference.py --input input_video.mov --prompt "your prompt here"

# 배치 처리
bash batch_video_with_prompts.sh
```

#### 영상 전처리

```bash
# 81프레임 단위로 영상 분할
python video_splitter.py input_video.mov -o ./inputs/

# 여러 영상 일괄 분할
for video in ./pre_processed_files/*.mov; do 
    python video_splitter.py "$video" -o ./inputs/
done
```

#### 프롬프트 파일 설정

`video_prompts.txt` 파일에 다음 형식으로 작성:

```
video_00000024_chunk1.mov|dark room, no lights, no people
video_00000057_chunk1.mov|The person in the middle is giving a speech alone in a dark room
video_00000184_chunk1.mov|A cloudless sky
video_00000227_chunk1.mov|People waving blue flags and cheering
video_00000370_chunk1.mov|
```

#### 다중 GPU 병렬처리

```bash
# 다중 GPU 환경에서 실행
CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py --multi_gpu
```

### 3. 파일 관리

#### 파일명 정규화

한글 파일명 문제 해결을 위한 영문 변환:

```bash
for file in *.mov; do
    number=$(echo "$file" | grep -o '[0-9]\{8\}' | head -1)
    chunk=$(echo "$file" | grep -o 'chunk[0-9]')
    new_name="video_${number}_${chunk}.mov"
    mv "$file" "$new_name"
done
```

## 📁 디렉토리 구조

```
├── inputs/                 # 처리할 영상 파일들 (81프레임 청크)
├── outputs/               # 결과물 저장 디렉토리
├── pre_processed_files/   # 전처리 대상 원본 영상들
├── video_splitter.py     # 영상 분할 도구
├── batch_video_with_prompts.sh  # 배치 처리 스크립트
├── video_prompts.txt     # 프롬프트 설정 파일
└── inference.py          # 메인 추론 스크립트
```

## ⚠️ 알려진 제약사항

1. **메모리 제한**: 3초 이상 영상 처리 시 메모리 부족 문제 발생 ([관련 이슈](https://github.com/ali-vilab/VACE/issues/56))
2. **프레임 정확성**: 입력 영상이 정확히 81프레임이 아닐 경우 결과물 프레임 수 불일치 가능
3. **한글 파일명**: 한글 파일명 인식 불가 (영문 변환 필요)

## 🔧 문제 해결

### 메모리 부족
다중 GPU 환경 구성 또는 영상 길이 단축 필요.

### 프레임 불일치
`video_splitter.py`를 통한 81프레임 패딩 적용.

## 📋 TODO

### 높은 우선순위
- [ ] **영상 캡셔닝 도구**: 자동 프롬프트 생성 (씬별 10개 프롬프트)
- [ ] **신별 영상 분할 개선**: 현재 불완전한 분할 로직 보완
- [ ] **프레임 일관성 검증**: 원본과 결과물 프레임 수 비교 도구

### 중간 우선순위  
- [ ] **문서화 완성**: 파이프라인 전체 과정 상세 설명
- [ ] **UI 개선**: 기존 UI를 현재 워크플로우에 맞게 수정
- [ ] **모델 최적화**: CLIP 모델 캐싱 및 메모리 최적화

### 낮은 우선순위
- [ ] **결과물 디렉토리명 개선**: 시간 기반 → 파일명 기반
- [ ] **Wan 14B quantization**: Single A100에서 실행 가능성 검토

## 🤝 기여하기

1. 이 repository를 fork
2. feature 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 push (`git push origin feature/amazing-feature`)
5. Pull Request 생성