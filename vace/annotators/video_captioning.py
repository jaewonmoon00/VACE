# /data/VACE/vace/annotators/video_captioning.py
# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import cv2
import os
import logging
import base64
import requests
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import tempfile
from typing import Optional, Union
import numpy as np

class VideoCaptioning:
    """
    단일 장면 비디오 자동 캡셔닝 클래스
    중간 프레임 추출 + 이미지 캡셔닝 방식
    """
    
    def __init__(self, method="blip2", device="auto"):
        """
        Args:
            method: 캡셔닝 방법 ("blip2", "gpt4v")
            device: 로컬 모델 사용시 디바이스 ("auto", "cuda", "cpu")
        """
        self.method = method
        
        # 디바이스 자동 선택
        if device == "auto":
            if torch.cuda.is_available():
                self.device = f"cuda:{torch.cuda.current_device()}"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # 모델 초기화
        self.processor = None
        self.model = None
        
        if method == "blip2":
            self._init_blip2()
        
        logging.info(f"VideoCaptioning initialized - method: {method}, device: {self.device}")
    
    def _init_blip2(self):
        """BLIP-2 모델 초기화"""
        try:
            logging.info("Loading BLIP-2 model...")
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            # GPU 사용 가능하면 이동
            if "cuda" in self.device:
                self.model = self.model.to(self.device)
                # 메모리 절약을 위해 half precision 사용
                if torch.cuda.is_available():
                    self.model = self.model.half()
            
            self.model.eval()  # 평가 모드
            logging.info("BLIP-2 model loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to load BLIP-2 model: {e}")
            self.processor = None
            self.model = None
            raise
    
    def extract_middle_frame(self, video_path: str) -> Image.Image:
        """
        비디오에서 중간 프레임 추출
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            PIL Image: 중간 프레임 이미지
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        try:
            # 총 프레임 수 확인
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                raise ValueError(f"Video has no frames: {video_path}")
            
            # 중간 프레임으로 이동
            middle_frame_idx = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
            
            # 프레임 읽기
            ret, frame = cap.read()
            
            if not ret:
                raise ValueError(f"Failed to read middle frame from: {video_path}")
            
            # OpenCV (BGR) -> PIL (RGB) 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            logging.info(f"Extracted middle frame ({middle_frame_idx}/{total_frames}) from {video_path}")
            return pil_image
            
        finally:
            cap.release()
    
    def caption_with_blip2(self, image: Image.Image) -> str:
        """
        BLIP-2 모델로 이미지 캡셔닝
        
        Args:
            image: PIL Image
            
        Returns:
            str: 생성된 캡션
        """
        if self.processor is None or self.model is None:
            return "BLIP-2 model not initialized"
        
        try:
            # 이미지 크기 조정 (메모리 절약)
            if image.size[0] > 512 or image.size[1] > 512:
                image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            
            # 이미지 전처리
            inputs = self.processor(image, return_tensors="pt")
            
            # GPU로 이동
            if "cuda" in self.device:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                if torch.cuda.is_available():
                    inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
            # 캡션 생성
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_length=50, 
                    num_beams=5,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
            
            # 디코딩
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
            logging.info(f"BLIP-2 generated caption: {caption}")
            return caption
            
        except Exception as e:
            logging.error(f"BLIP-2 captioning failed: {e}")
            return f"Failed to generate caption: {str(e)}"
    
    def caption_with_gpt4v(self, image: Image.Image, api_key: str) -> str:
        """
        GPT-4V API로 이미지 캡셔닝
        
        Args:
            image: PIL Image
            api_key: OpenAI API 키
            
        Returns:
            str: 생성된 캡션
        """
        try:
            # 이미지를 base64로 인코딩
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                # 이미지 크기 조정 (API 비용 절약)
                if image.size[0] > 1024 or image.size[1] > 1024:
                    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                
                image.save(tmp_file.name, "JPEG", quality=85)
                
                with open(tmp_file.name, "rb") as img_file:
                    base64_image = base64.b64encode(img_file.read()).decode('utf-8')
                
                os.unlink(tmp_file.name)
            
            # API 요청
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Based on this video frame, imagine what scenes might exist to the left and right. "
            "Do not focus on describing specific objects or people. Instead, emphasize the overall atmosphere, "
            "lighting, color palette, and cinematic style of the surrounding environment. "
            "Describe it like a film director setting the tone for a wide, immersive shot. "
            "Keep the description under 100 words."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"  # 비용 절약
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.3
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                caption = response.json()["choices"][0]["message"]["content"]
                logging.info(f"GPT-4V generated caption: {caption}")
                return caption
            else:
                error_msg = response.json().get("error", {}).get("message", "Unknown error")
                logging.error(f"GPT-4V API error: {response.status_code} - {error_msg}")
                return f"GPT-4V API error: {error_msg}"
                
        except Exception as e:
            logging.error(f"GPT-4V captioning failed: {e}")
            return f"Failed to generate caption with GPT-4V: {str(e)}"
    
    def caption_video(self, video_path: str, api_key: Optional[str] = None) -> str:
        """
        비디오 캡셔닝 메인 함수
        
        Args:
            video_path: 비디오 파일 경로
            api_key: API 키 (API 방식 사용시)
            
        Returns:
            str: 생성된 캡션
        """
        try:
            # 1. 중간 프레임 추출
            middle_frame = self.extract_middle_frame(video_path)
            
            # 2. 선택된 방법으로 캡셔닝
            if self.method == "blip2":
                caption = self.caption_with_blip2(middle_frame)
            elif self.method == "gpt4v":
                if not api_key:
                    raise ValueError("API key required for GPT-4V method")
                caption = self.caption_with_gpt4v(middle_frame, api_key)
            else:
                raise ValueError(f"Unsupported captioning method: {self.method}")
            
            # 3. 후처리
            caption = self._post_process_caption(caption)
            
            return caption
            
        except Exception as e:
            logging.error(f"Video captioning failed: {e}")
            return f"Error: {str(e)}"
    
    def _post_process_caption(self, caption: str) -> str:
        """
        캡션 후처리 (정제, 형식 통일 등)
        
        Args:
            caption: 원본 캡션
            
        Returns:
            str: 후처리된 캡션
        """
        if not caption or caption.startswith(("Error:", "Failed", "GPT-4V API error:")):
            return caption
        
        # 기본적인 정제
        caption = caption.strip()
        
        # 첫 글자 대문자화
        if caption and not caption[0].isupper():
            caption = caption[0].upper() + caption[1:]
        
        # 마침표 추가 (없는 경우)
        if caption and not caption.endswith(('.', '!', '?')):
            caption += '.'
        
        return caption
    
    def get_model_info(self) -> dict:
        """모델 정보 반환"""
        return {
            "method": self.method,
            "device": self.device,
            "model_loaded": self.model is not None,
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        }

# 유틸리티 함수들
def check_requirements():
    """필요한 패키지 확인"""
    requirements = {
        "cv2": "opencv-python",
        "PIL": "Pillow", 
        "torch": "torch",
        "transformers": "transformers",
        "requests": "requests"
    }
    
    missing = []
    for module, package in requirements.items():
        try:
            if module == "cv2":
                import cv2
            elif module == "PIL":
                from PIL import Image
            else:
                __import__(module)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing.append(package)
    
    if missing:
        print(f"\n🔧 Install missing packages:")
        print(f"pip install {' '.join(missing)}")
        return False
    else:
        print("\n🎉 All requirements satisfied!")
        return True

# 테스트 함수
def test_captioning(video_path: str, method: str = "blip2"):
    """캡셔닝 테스트"""
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    try:
        captioner = VideoCaptioning(method=method)
        print(f"🎬 Testing {method} captioning...")
        
        caption = captioner.caption_video(video_path)
        print(f"📝 Generated caption: {caption}")
        
        # 모델 정보 출력
        info = captioner.get_model_info()
        print(f"🔧 Model info: {info}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    # 요구사항 체크
    print("🔍 Checking requirements...")
    if check_requirements():
        # 테스트 실행 (실제 비디오 경로로 변경)
        test_video = "/data/VACE/assets/videos/test.mp4"
        if os.path.exists(test_video):
            test_captioning(test_video)
        else:
            print(f"📁 Test video not found: {test_video}")
            print("💡 Place a test video and run: python video_captioning.py")