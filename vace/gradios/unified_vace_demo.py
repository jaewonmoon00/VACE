# /data/VACE/vace/gradios/unified_vace_demo.py

import gradio as gr
import os
import sys
import time
import threading
import random
from pathlib import Path
import subprocess
import torch
import shutil
import logging

# 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# 🆕 비디오 캡셔닝 모듈 import
try:
    from vace.annotators.video_captioning import VideoCaptioning
    CAPTIONING_AVAILABLE = True
    print("✅ Video captioning feature enabled")
except ImportError as e:
    print(f"⚠️ Video captioning not available: {e}")
    print("💡 To enable: pip install transformers torch torchvision")
    CAPTIONING_AVAILABLE = False


class UnifiedVACEDemo:
    def __init__(self, cfg):
        self.cfg = cfg
        self.batch_stop_flag = False
        self.batch_thread = None
        
        # 🆕 GPU 정보 감지
        self.available_gpus = self._detect_gpus()
        print(f"🔍 Detected GPUs: {self.available_gpus}")
        
        # 🆕 비디오 캡셔닝 모듈 초기화
        self.video_captioner = None
        if CAPTIONING_AVAILABLE:
            self._init_video_captioning()
    
    def _detect_gpus(self):
        """사용 가능한 GPU 감지"""
        gpu_info = []
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    memory = torch.cuda.get_device_properties(i).total_memory // (1024**3)  # GB
                    gpu_info.append({
                        'id': i,
                        'name': gpu_name,
                        'memory': memory
                    })
            else:
                gpu_info.append({'id': -1, 'name': 'CPU Only', 'memory': 0})
        except Exception as e:
            print(f"GPU detection error: {e}")
            gpu_info.append({'id': -1, 'name': 'CPU Only', 'memory': 0})
        
        return gpu_info
    
    def _get_gpu_choices(self):
        """GPU 선택 옵션 생성"""
        choices = []
        if len(self.available_gpus) > 1 and self.available_gpus[0]['id'] != -1:
            # 다중 GPU 옵션
            choices.append("Auto (All Available GPUs)")
            choices.append("Multi-GPU (Custom)")
            choices.append("---")  # 구분선
        
        # 개별 GPU 옵션
        for gpu in self.available_gpus:
            if gpu['id'] == -1:
                choices.append("CPU Only")
            else:
                choices.append(f"GPU {gpu['id']}: {gpu['name']} ({gpu['memory']}GB)")
        
        return choices
    
    def _parse_gpu_selection(self, gpu_choice):
        """GPU 선택 파싱"""
        if gpu_choice == "Auto (All Available GPUs)":
            return "auto", None
        elif gpu_choice == "Multi-GPU (Custom)":
            return "multi", None
        elif gpu_choice == "CPU Only":
            return "cpu", None
        elif gpu_choice.startswith("GPU "):
            gpu_id = int(gpu_choice.split(":")[0].split()[1])
            return "single", gpu_id
        else:
            return "auto", None
    
    def _get_execution_config(self, gpu_mode, gpu_id, custom_gpu_ids=None):
        """실행 설정 생성"""
        config = {
            'env': {},
            'nproc_per_node': 1,
            'use_torchrun': False,
            'cuda_visible_devices': None
        }
        
        if gpu_mode == "single" and gpu_id is not None:
            # 단일 GPU 실행
            config['cuda_visible_devices'] = str(gpu_id)
            config['nproc_per_node'] = 1
            config['use_torchrun'] = False
            config['env']['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            
        elif gpu_mode == "multi":
            # 다중 GPU 실행
            if custom_gpu_ids:
                gpu_list = [str(i) for i in custom_gpu_ids]
                config['cuda_visible_devices'] = ','.join(gpu_list)
                config['nproc_per_node'] = len(custom_gpu_ids)
                config['env']['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_list)
            else:
                # 모든 GPU 사용
                gpu_list = [str(gpu['id']) for gpu in self.available_gpus if gpu['id'] != -1]
                config['cuda_visible_devices'] = ','.join(gpu_list)
                config['nproc_per_node'] = len(gpu_list)
                if gpu_list:
                    config['env']['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_list)
            
            config['use_torchrun'] = config['nproc_per_node'] > 1
            
        elif gpu_mode == "auto":
            # 자동 선택 (기존 로직)
            available_gpu_count = len([g for g in self.available_gpus if g['id'] != -1])
            if available_gpu_count > 1:
                config['nproc_per_node'] = min(available_gpu_count, 2)
                config['use_torchrun'] = True
            else:
                config['nproc_per_node'] = 1
                config['use_torchrun'] = False
                if available_gpu_count == 1:
                    config['env']['CUDA_VISIBLE_DEVICES'] = '0'
        
        elif gpu_mode == "cpu":
            # CPU 전용
            config['env']['CUDA_VISIBLE_DEVICES'] = ""
            config['nproc_per_node'] = 1
            config['use_torchrun'] = False
        
        return config

    def _init_video_captioning(self):
        """비디오 캡셔닝 모듈 초기화"""
        try:
            # 기본적으로 BLIP-2 사용 (로컬 처리)
            self.video_captioner = VideoCaptioning(method="blip2", device="auto")
            logging.info("Video captioning module initialized successfully")
        except Exception as e:
            logging.warning(f"Failed to initialize video captioning: {e}")
            self.video_captioner = None

    # 🆕 랜덤 시드 생성 함수들
    def generate_random_seed(self):
        """랜덤 시드 생성 (0 ~ 2^32-1 범위)"""
        return random.randint(0, 2**32 - 1)
    
    def randomize_pipeline_seed(self):
        """파이프라인 탭 시드 랜덤화"""
        return self.generate_random_seed()
    
    def randomize_sequential_seed(self):
        """순차 확장 탭 시드 랜덤화"""
        return self.generate_random_seed()
    
    def randomize_partial_seed(self):
        """부분 재생성 탭 시드 랜덤화"""
        return self.generate_random_seed()
    
    def randomize_batch_seed(self):
        """배치 처리용 시드 랜덤화"""
        return self.generate_random_seed()

    def create_gpu_settings_ui(self, tab_prefix=""):
        """GPU 설정 UI 컴포넌트 생성"""
        gpu_choices = self._get_gpu_choices()
        
        with gr.Accordion("🎛️ GPU Settings", open=False):
            gpu_selection = gr.Dropdown(
                choices=gpu_choices,
                value=gpu_choices[0] if gpu_choices else "CPU Only",
                label="GPU Selection",
                elem_id=f"{tab_prefix}_gpu_selection"
            )
            
            # 커스텀 다중 GPU 설정 (조건부 표시)
            with gr.Row(visible=False) as custom_gpu_row:
                custom_gpu_ids = gr.CheckboxGroup(
                    choices=[f"GPU {gpu['id']}" for gpu in self.available_gpus if gpu['id'] != -1],
                    label="Select GPUs for Multi-GPU",
                    elem_id=f"{tab_prefix}_custom_gpu_ids"
                )
            
            # GPU 정보 표시
            gpu_info_text = self._format_gpu_info()
            gpu_info = gr.Markdown(
                value=gpu_info_text,
                label="Available GPUs"
            )
            
            # GPU 선택 변경 시 커스텀 설정 표시/숨김
            def toggle_custom_gpu_visibility(gpu_choice):
                return gr.update(visible=(gpu_choice == "Multi-GPU (Custom)"))
            
            gpu_selection.change(
                toggle_custom_gpu_visibility,
                inputs=[gpu_selection],
                outputs=[custom_gpu_row]
            )
        
        return gpu_selection, custom_gpu_ids
    
    def _format_gpu_info(self):
        """GPU 정보 포맷팅"""
        if not self.available_gpus or self.available_gpus[0]['id'] == -1:
            return "**Available Hardware:** CPU Only"
        
        info_lines = ["**Available GPUs:**"]
        for gpu in self.available_gpus:
            if gpu['id'] != -1:
                info_lines.append(f"- **GPU {gpu['id']}:** {gpu['name']} ({gpu['memory']}GB)")
        
        return "\n".join(info_lines)
            
    def create_ui(self):
        with gr.Blocks(title="Video Extender") as demo:
            gr.Markdown("""
            <div style="text-align: center; font-size: 28px; font-weight: bold; margin-bottom: 20px;">
                🎬 Freewillusion Video Extender
            </div>
            <div style="text-align: center; font-size: 16px; color: #666; margin-bottom: 20px;">
                Extend your videos with AI-powered outpainting
            </div>
            """)
            
            with gr.Tabs():
                # Tab 1: 메인 처리
                with gr.TabItem("🚀 Video Processing"):
                    self.create_pipeline_ui()
                
                # Tab 2: 배치 처리
                with gr.TabItem("📦 Batch Processing"):
                    self.create_batch_ui()
                
                # Tab 3: 순차적 영상 확장
                with gr.TabItem("🔗 Sequential Extension"):
                    self.create_sequential_ui()
                
                # Tab 4: 부분 재생성
                with gr.TabItem("🎯 Partial Regeneration"):
                    self.create_partial_ui()
            
        return demo
    
    def create_pipeline_ui(self):
        """기본 파이프라인 UI - GPU 설정 포함"""
        gr.Markdown("### 🎬 Video Extension")
        gr.Markdown("Upload a video and extend it with AI-powered outpainting")
    
        with gr.Row():
            with gr.Column():
                # 입력
                self.pipeline_video = gr.Video(label="Input Video")
                
                # 🆕 GPU 설정 추가
                self.pipeline_gpu_selection, self.pipeline_custom_gpu_ids = self.create_gpu_settings_ui("pipeline")
                
                # 🆕 자동 캡셔닝 섹션 추가
                if CAPTIONING_AVAILABLE:
                    with gr.Accordion("🎬 Auto Captioning", open=False):
                        with gr.Row():
                            self.pipeline_auto_caption_btn = gr.Button(
                                "🎬 Generate Caption",
                                variant="secondary",
                                scale=2
                            )
                            self.pipeline_caption_method = gr.Dropdown(
                                choices=["BLIP-2 (Local)", "GPT-4V (API)"],
                                value="BLIP-2 (Local)",
                                label="Method",
                                scale=3
                            )
                        
                        # API 키 입력 (조건부 표시)
                        self.pipeline_api_key = gr.Textbox(
                            label="OpenAI API Key (for GPT-4V)",
                            type="password",
                            placeholder="sk-...",
                            visible=False
                        )
                        
                        # 캡셔닝 상태 표시
                        self.pipeline_caption_status = gr.Textbox(
                            label="Caption Status",
                            value="Ready to generate caption",
                            interactive=False,
                            max_lines=2
                        )
                
                # 기본 설정
                with gr.Accordion("Extension Settings", open=True):
                    self.pipeline_direction = gr.CheckboxGroup(
                        choices=["left", "right", "up", "down"],
                        value=["left", "right"],
                        label="Extension Direction"
                    )
                    self.pipeline_expand_ratio = gr.Slider(
                        minimum=0.1, maximum=2.0, value=1.6, step=0.1,
                        label="Extension Ratio"
                    )
                    self.pipeline_prompt = gr.Textbox(
                        label="Description",
                        placeholder="Describe what you want to see in the extended areas... (Use Auto Caption to generate automatically)",
                        lines=3,
                        info="💡 Tip: Use the Auto Caption feature above to automatically generate descriptions!"
                    )
                
                # 모델 선택
                self.pipeline_model = gr.Dropdown(
                    choices=["14B", "1.3B"],
                    value="14B",
                    label="Model"
                )
                
                # 해상도 선택
                self.pipeline_size = gr.Dropdown(
                    choices=["720p", "480p"],
                    value="720p",
                    label="Output Resolution"
                )
                
                # 고급 설정
                with gr.Accordion("Advanced Settings", open=False):
                    self.pipeline_use_prompt_extend = gr.Dropdown(
                        choices=["None", "English", "Chinese"],
                        value="None",
                        label="Prompt Enhancement"
                    )
                    # 🆕 시드 설정 - 랜덤 버튼 추가
                    with gr.Row():
                        self.pipeline_seed = gr.Number(
                            value=2025, 
                            label="Seed",
                            scale=4
                        )
                        self.pipeline_random_seed_btn = gr.Button(
                            "🎲 Random", 
                            variant="secondary",
                            scale=1,
                            size="sm"
                        )
                    
                    self.pipeline_sampling_steps = gr.Slider(
                        minimum=20, maximum=100, value=50, step=5,
                        label="Quality Steps"
                    )
                    self.pipeline_guide_scale = gr.Slider(
                        minimum=1.0, maximum=10.0, value=5.0, step=0.5,
                        label="Guidance Scale"
                    )

                self.pipeline_run_btn = gr.Button("🚀 Extend Video", variant="primary")
                
            with gr.Column():
                # 결과 표시
                self.pipeline_progress = gr.Textbox(
                    label="Progress", 
                    interactive=False,
                    max_lines=5,
                    placeholder="Ready to process..."
                )
                self.pipeline_result_video = gr.Video(label="Extended Video")
        
        # 🆕 랜덤 시드 버튼 이벤트 추가
        self.pipeline_random_seed_btn.click(
            self.randomize_pipeline_seed,
            outputs=[self.pipeline_seed]
        )
        
        # 🆕 이벤트 핸들러 설정
        if CAPTIONING_AVAILABLE:
            # 캡션 방법 변경시 API 키 필드 표시/숨김
            def toggle_api_key_visibility(caption_method):
                return gr.update(visible="GPT-4V" in caption_method)
            
            self.pipeline_caption_method.change(
                toggle_api_key_visibility,
                inputs=[self.pipeline_caption_method],
                outputs=[self.pipeline_api_key]
            )
            
            # 자동 캡셔닝 버튼 클릭 이벤트
            self.pipeline_auto_caption_btn.click(
                self.generate_auto_caption,
                inputs=[
                    self.pipeline_video,
                    self.pipeline_caption_method,
                    self.pipeline_api_key
                ],
                outputs=[
                    self.pipeline_prompt,
                    self.pipeline_caption_status
                ]
            )
        
        # 모델 변경 시 해상도 옵션 업데이트
        def update_size_options(model_choice):
            if model_choice == "14B":
                return gr.update(choices=["720p", "480p", "720×1280", "1280×720"], value="720p")
            elif model_choice == "1.3B":
                return gr.update(choices=["480p", "480×832", "832×480"], value="480p")
            return gr.update(choices=["720p", "480p"], value="720p")
        
        # 이벤트 핸들러
        self.pipeline_model.change(
            update_size_options,
            inputs=[self.pipeline_model],
            outputs=[self.pipeline_size]
        )

        self.pipeline_run_btn.click(
            self.run_full_pipeline,
            inputs=[
                self.pipeline_video, self.pipeline_direction, self.pipeline_expand_ratio,
                self.pipeline_prompt, self.pipeline_use_prompt_extend,
                self.pipeline_seed, self.pipeline_model, self.pipeline_size,
                self.pipeline_sampling_steps, self.pipeline_guide_scale,
                self.pipeline_gpu_selection, self.pipeline_custom_gpu_ids  # 🆕 GPU 설정 추가
            ],
            outputs=[
                self.pipeline_progress, self.pipeline_result_video
            ]
        )
    
    def create_batch_ui(self):
        """배치 처리 UI - GPU 설정 포함"""
        gr.Markdown("### 📦 Batch Video Processing")
        gr.Markdown("Process multiple videos automatically.")
        
        with gr.Row():
            with gr.Column():
                self.batch_input_dir = gr.Textbox(
                    value="inputs/",
                    label="Input Directory",
                    placeholder="Path to directory containing videos"
                )
                self.batch_prompt_file = gr.File(
                    label="Prompt File (Optional)",
                    file_types=[".txt"]
                )
                
                # 🆕 GPU 설정 추가
                self.batch_gpu_selection, self.batch_custom_gpu_ids = self.create_gpu_settings_ui("batch")
                
                # 배치 설정
                with gr.Accordion("Batch Settings", open=True):
                    self.batch_direction = gr.CheckboxGroup(
                        choices=["left", "right", "up", "down"],
                        value=["left", "right"],
                        label="Extension Direction"
                    )
                    self.batch_expand_ratio = gr.Slider(
                        minimum=0.1, maximum=2.0, value=1.6,
                        label="Extension Ratio"
                    )
                    self.batch_model = gr.Dropdown(
                        choices=["14B", "1.3B"],
                        value="14B",
                        label="Model"
                    )
                    self.batch_size = gr.Dropdown(
                        choices=["720p", "480p"],
                        value="720p",
                        label="Output Resolution"
                    )
                    
                    # 🆕 배치용 시드 설정
                    with gr.Row():
                        self.batch_base_seed = gr.Number(
                            value=2025,
                            label="Base Seed",
                            scale=4,
                            info="Each video will use base_seed + index"
                        )
                        self.batch_random_seed_btn = gr.Button(
                            "🎲 Random", 
                            variant="secondary",
                            scale=1,
                            size="sm"
                        )
                
                with gr.Row():
                    self.batch_start_btn = gr.Button("▶️ Start Batch", variant="primary")
                    self.batch_stop_btn = gr.Button("⏹️ Stop")
                
                self.batch_progress = gr.Textbox(
                    label="Batch Progress",
                    max_lines=10,
                    interactive=False,
                    placeholder="Ready to start batch processing..."
                )
            
            with gr.Column():
                self.batch_results = gr.Gallery(
                    label="Batch Results",
                    columns=2,
                    rows=3
                )
                self.batch_status = gr.JSON(
                    label="Processing Status",
                    value={"total": 0, "completed": 0, "failed": 0, "current": ""}
                )
        
        # 🆕 배치 랜덤 시드 버튼 이벤트 추가
        self.batch_random_seed_btn.click(
            self.randomize_batch_seed,
            outputs=[self.batch_base_seed]
        )
        
        # 배치 모델 변경 시 해상도 옵션 업데이트
        def update_batch_size_options(model_choice):
            if model_choice == "14B":
                return gr.update(choices=["720p", "480p", "720×1280", "1280×720"], value="720p")
            elif model_choice == "1.3B":
                return gr.update(choices=["480p", "480×832", "832×480"], value="480p")
            return gr.update(choices=["720p", "480p"], value="720p")
        
        # 이벤트 핸들러
        self.batch_model.change(
            update_batch_size_options,
            inputs=[self.batch_model],
            outputs=[self.batch_size]
        )
        
        self.batch_start_btn.click(
            self.start_batch_processing,
            inputs=[
                self.batch_input_dir, self.batch_prompt_file,
                self.batch_direction, self.batch_expand_ratio, self.batch_model, self.batch_size,
                self.batch_base_seed,  # 🆕 배치 시드 추가
                self.batch_gpu_selection, self.batch_custom_gpu_ids  # 🆕 GPU 설정 추가
            ],
            outputs=[self.batch_progress, self.batch_status]
        )
        
        self.batch_stop_btn.click(
            self.stop_batch_processing,
            outputs=[self.batch_progress]
        )

    def create_sequential_ui(self):
        """순차적 영상 확장 UI - GPU 설정 및 캡셔닝 기능 포함"""
        gr.Markdown("### 🔗 Sequential Video Extension")
        gr.Markdown("""
        **Purpose**: Create longer videos by processing them in segments with seamless transitions.
        
        **How it works**: 
        - Takes the last 40 frames from the previous segment
        - Combines with first 41 frames of current segment  
        - Ensures smooth continuity between segments
        """)
        
        with gr.Row():
            with gr.Column():
                # 입력 설정
                with gr.Accordion("Input Settings", open=True):
                    self.seq_current_video = gr.Video(label="Current Segment Video")
                    self.seq_previous_video = gr.Video(label="Previous Segment Result")
                    
                    # 🆕 자동 캡셔닝 추가
                    if CAPTIONING_AVAILABLE:
                        with gr.Row():
                            self.seq_auto_caption_btn = gr.Button(
                                "🎬 Caption Current", 
                                variant="secondary",
                                scale=2
                            )
                            self.seq_caption_method = gr.Dropdown(
                                choices=["BLIP-2 (Local)", "GPT-4V (API)"],
                                value="BLIP-2 (Local)",
                                label="Method",
                                scale=2
                            )
                
                # 🆕 GPU 설정 추가
                self.seq_gpu_selection, self.seq_custom_gpu_ids = self.create_gpu_settings_ui("seq")
                
                # 연결 설정
                with gr.Accordion("Sequence Settings", open=True):
                    self.seq_front_frames = gr.Slider(
                        minimum=20, maximum=60, value=41, step=1,
                        label="Front Frames (from current)"
                    )
                    self.seq_insert_frames = gr.Slider(
                        minimum=20, maximum=60, value=40, step=1,
                        label="Insert Frames (from previous)"
                    )
                    
                # 생성 설정
                with gr.Accordion("Generation Settings", open=True):
                    self.seq_prompt = gr.Textbox(
                        label="Description",
                        placeholder="Describe the content for this segment... (or use Auto Caption)",
                        lines=3
                    )
                    self.seq_model = gr.Dropdown(
                        choices=["14B", "1.3B"],
                        value="14B",
                        label="Model"
                    )
                    self.seq_size = gr.Dropdown(
                        choices=["720p", "480p"],
                        value="720p", 
                        label="Output Resolution"
                    )
                    
                # 고급 설정
                with gr.Accordion("Advanced Settings", open=False):
                    # 🆕 순차 확장용 시드 설정
                    with gr.Row():
                        self.seq_seed = gr.Number(
                            value=2025, 
                            label="Seed",
                            scale=4
                        )
                        self.seq_random_seed_btn = gr.Button(
                            "🎲 Random", 
                            variant="secondary",
                            scale=1,
                            size="sm"
                        )
                    
                    self.seq_sampling_steps = gr.Slider(
                        minimum=20, maximum=100, value=50, step=5,
                        label="Quality Steps"
                    )
                    self.seq_guide_scale = gr.Slider(
                        minimum=1.0, maximum=10.0, value=5.0, step=0.5,
                        label="Guidance Scale"
                    )
                
                self.seq_run_btn = gr.Button("🔗 Generate Sequential Segment", variant="primary")
                
            with gr.Column():
                # 결과 및 진행상황
                self.seq_progress = gr.Textbox(
                    label="Progress",
                    interactive=False,
                    max_lines=8,
                    placeholder="Ready to process sequential segment..."
                )
                self.seq_result_video = gr.Video(label="Sequential Result")
                
                # 시각적 가이드
                gr.Markdown("""
                ### 📊 Processing Flow:
                1. **Extract**: Last N frames from previous video
                2. **Combine**: With first M frames of current video  
                3. **Generate**: Seamless continuation
                4. **Result**: Naturally connected segment
                
                **Tip**: Use this for creating videos longer than 81 frames (3.4 seconds at 24fps)
                """)

        # 🆕 순차 확장 랜덤 시드 버튼 이벤트 추가
        self.seq_random_seed_btn.click(
            self.randomize_sequential_seed,
            outputs=[self.seq_seed]
        )

        # 순차 확장용 캡셔닝 이벤트
        if CAPTIONING_AVAILABLE:
            self.seq_auto_caption_btn.click(
                self.generate_auto_caption,
                inputs=[
                    self.seq_current_video,
                    self.seq_caption_method,
                    self.pipeline_api_key  # API 키 공유
                ],
                outputs=[
                    self.seq_prompt,
                    self.pipeline_caption_status  # 상태 공유
                ]
            )

        # 모델 변경 시 해상도 업데이트
        def update_seq_size_options(model_choice):
            if model_choice == "14B":
                return gr.update(choices=["720p", "480p", "720×1280", "1280×720"], value="720p")
            elif model_choice == "1.3B":
                return gr.update(choices=["480p", "480×832", "832×480"], value="480p")
            return gr.update(choices=["720p", "480p"], value="720p")
        
        # 이벤트 핸들러
        self.seq_model.change(
            update_seq_size_options,
            inputs=[self.seq_model],
            outputs=[self.seq_size]
        )
        
        self.seq_run_btn.click(
            self.run_sequential_extension,
            inputs=[
                self.seq_current_video, self.seq_previous_video,
                self.seq_front_frames, self.seq_insert_frames,
                self.seq_prompt, self.seq_model, self.seq_size,
                self.seq_seed, self.seq_sampling_steps, self.seq_guide_scale,
                self.seq_gpu_selection, self.seq_custom_gpu_ids  # 🆕 GPU 설정 추가
            ],
            outputs=[self.seq_progress, self.seq_result_video]
        )

    def create_partial_ui(self):
        """부분 재생성 UI - GPU 설정 및 캡셔닝 기능 포함"""
        gr.Markdown("### 🎯 Partial Video Regeneration") 
        gr.Markdown("""
        **Purpose**: Keep the good parts of a video and regenerate only the unsatisfactory portions.
        
        **How it works**:
        - Use early frames as guidance (mask = 0, no generation)
        - Regenerate later frames while maintaining continuity
        - Perfect for fixing problematic endings while keeping good beginnings
        """)
        
        with gr.Row():
            with gr.Column():
                # 입력 설정
                with gr.Accordion("Input Settings", open=True):
                    self.partial_source_video = gr.Video(label="Source Video (to fix)")
                    
                    # 🆕 자동 캡셔닝 추가
                    if CAPTIONING_AVAILABLE:
                        with gr.Row():
                            self.partial_auto_caption_btn = gr.Button(
                                "🎬 Caption Source", 
                                variant="secondary",
                                scale=2
                            )
                            self.partial_caption_method = gr.Dropdown(
                                choices=["BLIP-2 (Local)", "GPT-4V (API)"],
                                value="BLIP-2 (Local)",
                                label="Method",
                                scale=2
                            )
                
                # 🆕 GPU 설정 추가
                self.partial_gpu_selection, self.partial_custom_gpu_ids = self.create_gpu_settings_ui("partial")
                
                # 가이드 설정
                with gr.Accordion("Guidance Settings", open=True):
                    self.partial_target_fps = gr.Slider(
                        minimum=16, maximum=30, value=24, step=2,
                        label="Target FPS"
                    )
                    self.partial_guide_duration = gr.Slider(
                        minimum=1/24,  # 1 frame at 24fps
                        maximum=5.0, 
                        value=2.6, 
                        step=1/24,
                        label="Guide Duration (seconds)",
                        info="Minimum: 1 frame, adjusts automatically with FPS"
                    )
                    
                # 재생성 설정  
                with gr.Accordion("Regeneration Settings", open=True):
                    self.partial_prompt = gr.Textbox(
                        label="New Description",
                        placeholder="Describe what you want in the regenerated portion... (or use Auto Caption)",
                        lines=3,
                        value="",
                    )
                    self.partial_model = gr.Dropdown(
                        choices=["14B", "1.3B"],
                        value="14B",
                        label="Model"
                    )
                    self.partial_size = gr.Dropdown(
                        choices=["720p", "480p"],
                        value="720p",
                        label="Output Resolution"
                    )
                    
                # 고급 설정
                with gr.Accordion("Advanced Settings", open=False):
                    # 🆕 부분 재생성용 시드 설정
                    with gr.Row():
                        self.partial_seed = gr.Number(
                            value=2025, 
                            label="Seed",
                            scale=4
                        )
                        self.partial_random_seed_btn = gr.Button(
                            "🎲 Random", 
                            variant="secondary",
                            scale=1,
                            size="sm"
                        )
                    
                    self.partial_sampling_steps = gr.Slider(
                        minimum=20, maximum=100, value=50, step=5,
                        label="Quality Steps"
                    )
                    self.partial_guide_scale = gr.Slider(
                        minimum=1.0, maximum=10.0, value=5.0, step=0.5,
                        label="Guidance Scale"
                    )
                
                self.partial_run_btn = gr.Button("🎯 Regenerate Partial Video", variant="primary")
                
            with gr.Column():
                # 결과 및 진행상황
                self.partial_progress = gr.Textbox(
                    label="Progress",
                    interactive=False, 
                    max_lines=8,
                    placeholder="Ready to process partial regeneration..."
                )
                self.partial_result_video = gr.Video(label="Partially Regenerated Result")
                
                # 프레임 계산 표시
                self.partial_frame_info = gr.Markdown("""
                ### 📊 Frame Calculation:
                - **Guide frames**: Will be calculated based on duration and FPS
                - **Regeneration frames**: Remaining frames will be newly generated
                - **Total frames**: Typically 81 frames (3.4s at 24fps)
                
                **Example**: 2.6s × 24fps = 62 guide frames + 19 new frames
                """)

        # 🆕 부분 재생성 랜덤 시드 버튼 이벤트 추가
        self.partial_random_seed_btn.click(
            self.randomize_partial_seed,
            outputs=[self.partial_seed]
        )

        # 부분 재생성용 캡셔닝 이벤트
        if CAPTIONING_AVAILABLE:
            self.partial_auto_caption_btn.click(
                self.generate_auto_caption,
                inputs=[
                    self.partial_source_video,
                    self.partial_caption_method,
                    self.pipeline_api_key  # API 키 공유
                ],
                outputs=[
                    self.partial_prompt,
                    self.pipeline_caption_status  # 상태 공유
                ]
            )

        # 가이드 시간/FPS 변경 시 프레임 정보 업데이트
        def update_frame_info(duration, fps):
            """프레임 정보 업데이트"""
            guide_frames = int(duration * fps)
            total_frames = 81
            regen_frames = max(0, total_frames - guide_frames)
            
            status = "✅ Valid" if 1 <= guide_frames <= 80 else "⚠️ Invalid"
            
            info = f"""
            ### 📊 Frame Calculation:
            - **Guide frames**: {guide_frames} frames ({duration:.3f}s × {fps}fps)
            - **Regeneration frames**: {regen_frames} frames 
            - **Total frames**: {total_frames} frames
            
            **Status**: {status}
            """
            return info
        
        # 모델 변경 시 해상도 업데이트
        def update_partial_size_options(model_choice):
            if model_choice == "14B":
                return gr.update(choices=["720p", "480p", "720×1280", "1280×720"], value="720p")
            elif model_choice == "1.3B":
                return gr.update(choices=["480p", "480×832", "832×480"], value="480p")
            return gr.update(choices=["720p", "480p"], value="720p")
        def update_guide_duration_range(fps):
            """FPS 변경시 Guide Duration 범위와 스텝 업데이트"""
            min_duration = 1 / fps      # 1 프레임
            max_duration = 80 / fps     # 최대 80 프레임 (1프레임은 재생성용)
            step_size = 1 / fps         # 1 프레임 단위 스텝
            
            return gr.update(
                minimum=min_duration,
                maximum=max_duration,
                step=step_size,
                value=min(max(min_duration, 2.6), max_duration),  # 현재 값 조정
                info=f"Range: {min_duration:.3f}s - {max_duration:.3f}s (1-80 frames at {fps}fps)"
            )
        # 이벤트 핸들러
        self.partial_guide_duration.change(
            update_frame_info,
            inputs=[self.partial_guide_duration, self.partial_target_fps],
            outputs=[self.partial_frame_info]
        )
        
        self.partial_target_fps.change(
            update_guide_duration_range,
            inputs=[self.partial_target_fps],
            outputs=[self.partial_guide_duration]
        )

        self.partial_target_fps.change(
            update_frame_info,
            inputs=[self.partial_guide_duration, self.partial_target_fps],
            outputs=[self.partial_frame_info]
        )
        
        self.partial_model.change(
            update_partial_size_options,
            inputs=[self.partial_model],
            outputs=[self.partial_size]
        )
        
        self.partial_run_btn.click(
            self.run_partial_regeneration,
            inputs=[
                self.partial_source_video, self.partial_guide_duration, self.partial_target_fps,
                self.partial_prompt, self.partial_model, self.partial_size,
                self.partial_seed, self.partial_sampling_steps, self.partial_guide_scale,
                self.partial_gpu_selection, self.partial_custom_gpu_ids  # 🆕 GPU 설정 추가
            ],
            outputs=[self.partial_progress, self.partial_result_video]
        )

    # 🆕 자동 캡셔닝 함수
    def generate_auto_caption(self, video, caption_method, api_key):
        """
        자동 캡셔닝 함수
        
        Args:
            video: 업로드된 비디오 파일
            caption_method: 캡셔닝 방법
            api_key: API 키 (필요시)
            
        Returns:
            tuple: (생성된 캡션, 상태 메시지)
        """
        try:
            if not video:
                return "", "⚠️ Please upload a video first!"
            
            if not self.video_captioner:
                return "", "❌ Video captioning not available. Please check installation."
            
            # 진행 상태 표시
            status_msg = f"🔄 Generating caption with {caption_method}..."
            
            # 캡션 방법에 따라 설정
            if "BLIP-2" in caption_method:
                self.video_captioner.method = "blip2"
                caption = self.video_captioner.caption_video(video)
                
            elif "GPT-4V" in caption_method:
                if not api_key or not api_key.startswith("sk-"):
                    return "", "⚠️ Please provide a valid OpenAI API key for GPT-4V!"
                
                self.video_captioner.method = "gpt4v"
                caption = self.video_captioner.caption_video(video, api_key=api_key)
                
            else:
                return "", "❌ Unknown captioning method!"
            
            # 결과 확인
            if caption.startswith(("Error:", "Failed", "GPT-4V API error:")):
                return "", f"❌ {caption}"
            
            success_msg = f"✅ Caption generated successfully with {caption_method}"
            return caption, success_msg
            
        except Exception as e:
            error_msg = f"❌ Error generating caption: {str(e)}"
            logging.error(error_msg)
            return "", error_msg

    def _map_model_name(self, model_choice):
        """모델 선택을 실제 모델명으로 매핑"""
        mapping = {
            "14B": "vace-14B",
            "1.3B": "vace-1.3B"
        }
        return mapping.get(model_choice, "vace-14B")

    def _map_prompt_extension(self, extension_mode):
        """프롬프트 확장 모드 매핑"""
        mapping = {
            "None": "plain",
            "English": "wan_en", 
            "Chinese": "wan_zh"
        }
        return mapping.get(extension_mode, "plain")

    def run_full_pipeline(self, video, direction, expand_ratio, prompt, use_prompt_extend, 
                         seed, model_choice, size, sampling_steps, guide_scale, 
                         gpu_selection, custom_gpu_ids):
        """전체 파이프라인 실행 - GPU 설정 포함"""
        try:
            if not video:
                yield "❌ Please upload a video first!", None
                return

            model_name = self._map_model_name(model_choice)
            prompt_extend = self._map_prompt_extension(use_prompt_extend)
            task = "outpainting"

            # 🆕 GPU 설정 파싱
            gpu_mode, gpu_id = self._parse_gpu_selection(gpu_selection)
            
            # 커스텀 다중 GPU 처리
            selected_gpu_ids = None
            if gpu_mode == "multi" and custom_gpu_ids:
                selected_gpu_ids = [int(gpu.split()[1]) for gpu in custom_gpu_ids]
            
            # 실행 설정 생성
            exec_config = self._get_execution_config(gpu_mode, gpu_id, selected_gpu_ids)
            
            yield f"🔧 GPU Config: {gpu_selection} | Mode: {gpu_mode} | GPUs: {exec_config.get('cuda_visible_devices', 'Auto')}", None

            # 파라미터 처리
            try:
                seed = int(seed) if seed is not None else 2025
                sampling_steps = int(sampling_steps) if sampling_steps is not None else 50
                guide_scale = float(guide_scale) if guide_scale is not None else 5.0
                expand_ratio = float(expand_ratio) if expand_ratio is not None else 1.6
            except Exception as e:
                yield f"❌ Parameter error: {str(e)}", None
                return

            if size is None:
                size = "480p" if model_name == "vace-1.3B" else "720p"

            vace_root_dir = "/data/VACE"
            inputs_dir = os.path.join(vace_root_dir, "inputs")
            os.makedirs(inputs_dir, exist_ok=True)

            # 입력 비디오 복사
            original_name = Path(video).stem
            safe_filename = f"{original_name}.mp4"
            target_video_path = os.path.join(inputs_dir, safe_filename)
            try:
                shutil.copy2(video, target_video_path)
            except Exception as e:
                yield f"❌ File copy failed: {e}", None
                return

            # 디렉토리 설정
            timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
            pre_save_dir = os.path.join('/data/VACE/processed', task, f'{original_name}_{timestamp}')
            os.makedirs(pre_save_dir, exist_ok=True)
            result_folder_name = f"{original_name}-{task}-{timestamp}"
            result_save_dir = os.path.join(vace_root_dir, 'results', model_name, result_folder_name)

            # 모델별 체크포인트 디렉토리
            ckpt_dir_map = {
                "vace-14B": "models/Wan2.1-VACE-14B",
                "vace-1.3B": "models/Wan2.1-VACE-1.3B"
            }
            ckpt_dir = ckpt_dir_map.get(model_name, "models/Wan2.1-VACE-14B")

            # 1. Preprocess 단계
            preprocess_cmd = [
                'python', 'vace/vace_preproccess.py',
                '--task', str(task),
                '--direction', ','.join(direction) if direction else 'left,right',
                '--expand_ratio', str(expand_ratio),
                '--video', str(target_video_path),
                '--pre_save_dir', pre_save_dir
            ]
            yield "🔄 Preparing video...", None
            try:
                # 환경변수 적용
                preprocess_env = os.environ.copy()
                preprocess_env.update(exec_config['env'])
                
                result = subprocess.run(
                    preprocess_cmd,
                    capture_output=True,
                    text=True,
                    cwd=vace_root_dir,
                    env=preprocess_env
                )
                if result.returncode != 0:
                    error_msg = result.stderr if result.stderr else result.stdout
                    yield f"❌ Preparation failed:\n{error_msg}", None
                    return
            except Exception as e:
                yield f"❌ Preparation error: {str(e)}", None
                return

            # 2. Inference 단계
            src_video_path = os.path.join(pre_save_dir, f"src_video-{task}.mp4")
            src_mask_path = os.path.join(pre_save_dir, f"src_mask-{task}.mp4")

            # 기본 추론 명령어
            base_inference_cmd = [
                'vace/vace_wan_inference.py',
                '--src_video', src_video_path,
                '--src_mask', src_mask_path,
                '--base_seed', str(seed),
                '--model_name', str(model_name),
                '--ckpt_dir', str(ckpt_dir),
                '--size', str(size),
                '--sample_steps', str(sampling_steps),
                '--sample_guide_scale', str(guide_scale),
                '--save_dir', result_save_dir
            ]
            
            # 🆕 GPU 설정에 따른 명령어 구성
            if exec_config['use_torchrun']:
                inference_cmd = [
                    'torchrun',
                    f'--nproc-per-node={exec_config["nproc_per_node"]}',
                    '--master_port=12355'
                ] + base_inference_cmd
                
                # 다중 GPU 최적화 옵션
                if exec_config['nproc_per_node'] > 1:
                    inference_cmd.extend([
                        '--dit_fsdp',
                        '--t5_fsdp',
                        '--ulysses_size', str(exec_config['nproc_per_node']),
                        '--ring_size', '1'
                    ])
            else:
                # 단일 GPU 또는 CPU 실행
                inference_cmd = ['python'] + base_inference_cmd
                
            if prompt and prompt.strip():
                inference_cmd.extend(['--prompt', str(prompt)])
            else:
                inference_cmd.extend(['--prompt', ""])
            if prompt_extend != 'plain':
                inference_cmd.extend(['--use_prompt_extend', str(prompt_extend)])

            yield f"🚀 Extending video with {gpu_selection}...", None
            try:
                # 환경변수 적용
                inference_env = os.environ.copy()
                inference_env.update(exec_config['env'])
                
                result = subprocess.run(
                    inference_cmd,
                    capture_output=True,
                    text=True,
                    cwd=vace_root_dir,
                    env=inference_env
                )
                if result.returncode == 0:
                    out_video_path = os.path.join(result_save_dir, 'out_video.mp4')
                    if os.path.exists(out_video_path):
                        yield "✅ Video extension completed!", out_video_path
                    else:
                        yield "⚠️ Processing finished but result not found.", None
                else:
                    error_msg = result.stderr if result.stderr else result.stdout
                    yield f"❌ Extension failed:\n{error_msg}", None
            except Exception as e:
                yield f"❌ Extension error: {str(e)}", None
        except Exception as e:
            yield f"❌ Error: {str(e)}", None

    def run_sequential_extension(self, current_video, previous_video, front_frames, insert_frames,
                                prompt, model_choice, size, seed, sampling_steps, guide_scale,
                                gpu_selection, custom_gpu_ids):
        """순차적 영상 확장 실행 - GPU 설정 포함"""
        try:
            if not current_video:
                yield "❌ Please upload a current segment video!", None
                return
            
            if not previous_video:
                yield "❌ Please upload a previous segment video!", None
                return

            model_name = self._map_model_name(model_choice)
            
            # 🆕 GPU 설정 파싱
            gpu_mode, gpu_id = self._parse_gpu_selection(gpu_selection)
            selected_gpu_ids = None
            if gpu_mode == "multi" and custom_gpu_ids:
                selected_gpu_ids = [int(gpu.split()[1]) for gpu in custom_gpu_ids]
            exec_config = self._get_execution_config(gpu_mode, gpu_id, selected_gpu_ids)
            
            yield f"🔧 GPU Config: {gpu_selection} | Sequential processing...", None
            
            # 파라미터 검증
            try:
                front_frames = int(front_frames)
                insert_frames = int(insert_frames) 
                seed = int(seed) if seed is not None else 2025
                sampling_steps = int(sampling_steps) if sampling_steps is not None else 50
                guide_scale = float(guide_scale) if guide_scale is not None else 5.0
            except Exception as e:
                yield f"❌ Parameter error: {str(e)}", None
                return

            yield "🔄 Preparing sequential extension...", None

            vace_root_dir = "/data/VACE"
            inputs_dir = os.path.join(vace_root_dir, "inputs")
            os.makedirs(inputs_dir, exist_ok=True)

            # 파일 복사
            current_name = Path(current_video).stem
            previous_name = Path(previous_video).stem
            timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
            
            current_target = os.path.join(inputs_dir, f"{current_name}_current.mp4")
            previous_target = os.path.join(inputs_dir, f"{previous_name}_previous.mp4")
            
            shutil.copy2(current_video, current_target)
            shutil.copy2(previous_video, previous_target)

            # 결과 디렉토리
            result_folder_name = f"{current_name}-sequential-{timestamp}"
            result_save_dir = os.path.join(vace_root_dir, 'results', model_name, result_folder_name)
            os.makedirs(result_save_dir, exist_ok=True)

            # 모델별 체크포인트 디렉토리
            ckpt_dir_map = {
                "vace-14B": "models/Wan2.1-VACE-14B", 
                "vace-1.3B": "models/Wan2.1-VACE-1.3B"
            }
            ckpt_dir = ckpt_dir_map.get(model_name, "models/Wan2.1-VACE-14B")

            yield "🔗 Processing sequential connection...", None

            # 순차 연장용 기본 명령어
            base_inference_cmd = [
                'vace/vace_wan_inference.py',
                '--src_video', current_target,
                '--base_seed', str(seed),
                '--model_name', str(model_name),
                '--ckpt_dir', str(ckpt_dir),
                '--size', str(size),
                '--sample_steps', str(sampling_steps),
                '--sample_guide_scale', str(guide_scale),
                '--save_dir', result_save_dir,
                # 순차 연장 전용 파라미터
                '--sequential_mode', 'true',
                '--previous_video', previous_target,
                '--front_frames', str(front_frames),
                '--insert_frames', str(insert_frames)
            ]
            
            # 🆕 GPU 설정에 따른 명령어 구성
            if exec_config['use_torchrun']:
                inference_cmd = [
                    'torchrun',
                    f'--nproc-per-node={exec_config["nproc_per_node"]}',
                    '--master_port=12355'
                ] + base_inference_cmd
                
                if exec_config['nproc_per_node'] > 1:
                    inference_cmd.extend([
                        '--dit_fsdp',
                        '--t5_fsdp', 
                        '--ulysses_size', str(exec_config['nproc_per_node']),
                        '--ring_size', '1'
                    ])
            else:
                inference_cmd = ['python'] + base_inference_cmd
                
            if prompt and prompt.strip():
                inference_cmd.extend(['--prompt', str(prompt)])

            try:
                # 환경변수 적용
                inference_env = os.environ.copy()
                inference_env.update(exec_config['env'])
                
                result = subprocess.run(
                    inference_cmd,
                    capture_output=True,
                    text=True,
                    cwd=vace_root_dir,
                    env=inference_env
                )
                
                if result.returncode == 0:
                    out_video_path = os.path.join(result_save_dir, 'out_video.mp4')
                    if os.path.exists(out_video_path):
                        yield "✅ Sequential extension completed!", out_video_path
                    else:
                        yield "⚠️ Processing finished but result not found.", None
                else:
                    error_msg = result.stderr if result.stderr else result.stdout
                    yield f"❌ Sequential extension failed:\n{error_msg}", None
            except Exception as e:
                yield f"❌ Sequential extension error: {str(e)}", None

        except Exception as e:
            yield f"❌ Error: {str(e)}", None

    def run_partial_regeneration(self, source_video, guide_duration, target_fps, prompt, 
                               model_choice, size, seed, sampling_steps, guide_scale,
                               gpu_selection, custom_gpu_ids):
        """부분 재생성 실행 - GPU 설정 포함"""
        try:
            if not source_video:
                yield "❌ Please upload a source video!", None
                return

            model_name = self._map_model_name(model_choice)
            
            # 🆕 GPU 설정 파싱
            gpu_mode, gpu_id = self._parse_gpu_selection(gpu_selection)
            selected_gpu_ids = None
            if gpu_mode == "multi" and custom_gpu_ids:
                selected_gpu_ids = [int(gpu.split()[1]) for gpu in custom_gpu_ids]
            exec_config = self._get_execution_config(gpu_mode, gpu_id, selected_gpu_ids)
            
            # 파라미터 검증
            try:
                guide_duration = float(guide_duration)
                target_fps = int(target_fps)
                seed = int(seed) if seed is not None else 2025
                sampling_steps = int(sampling_steps) if sampling_steps is not None else 50
                guide_scale = float(guide_scale) if guide_scale is not None else 5.0
            except Exception as e:
                yield f"❌ Parameter error: {str(e)}", None
                return

            # 가이드 프레임 수 계산
            guide_frames = int(guide_duration * target_fps)
            total_frames = 81  # VACE 기본값
            
            if guide_frames >= total_frames:
                yield "❌ Guide duration too long! Must be shorter than total video length.", None
                return

            yield f"🔧 GPU: {gpu_selection} | Preparing partial regeneration (guide: {guide_frames} frames)...", None

            vace_root_dir = "/data/VACE"
            inputs_dir = os.path.join(vace_root_dir, "inputs")
            os.makedirs(inputs_dir, exist_ok=True)

            # 파일 복사
            source_name = Path(source_video).stem
            timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
            
            source_target = os.path.join(inputs_dir, f"{source_name}_source.mp4")
            shutil.copy2(source_video, source_target)

            # 결과 디렉토리
            result_folder_name = f"{source_name}-partial-{timestamp}"
            result_save_dir = os.path.join(vace_root_dir, 'results', model_name, result_folder_name)
            os.makedirs(result_save_dir, exist_ok=True)

            # 모델별 체크포인트 디렉토리
            ckpt_dir_map = {
                "vace-14B": "models/Wan2.1-VACE-14B",
                "vace-1.3B": "models/Wan2.1-VACE-1.3B"
            }
            ckpt_dir = ckpt_dir_map.get(model_name, "models/Wan2.1-VACE-14B")

            yield f"🎯 Processing partial regeneration ({total_frames - guide_frames} new frames)...", None

            # 부분 재생성용 기본 명령어
            base_inference_cmd = [
                'vace/vace_wan_inference.py',
                '--src_video', source_target,
                '--base_seed', str(seed),
                '--model_name', str(model_name),
                '--ckpt_dir', str(ckpt_dir),
                '--size', str(size),
                '--sample_steps', str(sampling_steps),
                '--sample_guide_scale', str(guide_scale),
                '--save_dir', result_save_dir,
                # 부분 재생성 전용 파라미터
                '--partial_mode', 'true',
                '--guide_duration', str(guide_duration),
                '--target_fps', str(target_fps)
            ]
            
            # 🆕 GPU 설정에 따른 명령어 구성
            if exec_config['use_torchrun']:
                inference_cmd = [
                    'torchrun',
                    f'--nproc-per-node={exec_config["nproc_per_node"]}',
                    '--master_port=12355'
                ] + base_inference_cmd
                
                if exec_config['nproc_per_node'] > 1:
                    inference_cmd.extend([
                        '--dit_fsdp',
                        '--t5_fsdp',
                        '--ulysses_size', str(exec_config['nproc_per_node']),
                        '--ring_size', '1'
                    ])
            else:
                inference_cmd = ['python'] + base_inference_cmd
                
            if prompt:
                inference_cmd.extend(['--prompt', str(prompt)])
            else:
                inference_cmd.extend(['--prompt', ""])

            try:
                # 환경변수 적용
                inference_env = os.environ.copy()
                inference_env.update(exec_config['env'])
                
                result = subprocess.run(
                    inference_cmd,
                    capture_output=True,
                    text=True,
                    cwd=vace_root_dir,
                    env=inference_env
                )
                
                if result.returncode == 0:
                    out_video_path = os.path.join(result_save_dir, 'out_video.mp4')
                    if os.path.exists(out_video_path):
                        regen_frames = total_frames - guide_frames
                        yield f"✅ Partial regeneration completed! ({regen_frames} frames regenerated)", out_video_path
                    else:
                        yield "⚠️ Processing finished but result not found.", None
                else:
                    error_msg = result.stderr if result.stderr else result.stdout
                    yield f"❌ Partial regeneration failed:\n{error_msg}", None
            except Exception as e:
                yield f"❌ Partial regeneration error: {str(e)}", None

        except Exception as e:
            yield f"❌ Error: {str(e)}", None
    
    def start_batch_processing(self, input_dir, prompt_file, direction, expand_ratio, model_choice, 
                             batch_size, base_seed, gpu_selection, custom_gpu_ids):
        """배치 처리 시작 - GPU 설정 포함"""
        if self.batch_thread and self.batch_thread.is_alive():
            return "⚠️ Batch processing is already running!", {"status": "running"}
        
        self.batch_stop_flag = False
        self.batch_thread = threading.Thread(
            target=self._batch_worker,
            args=(input_dir, prompt_file, direction, expand_ratio, model_choice, batch_size, 
                  base_seed, gpu_selection, custom_gpu_ids)
        )
        self.batch_thread.start()
        
        return "🚀 Batch processing started!", {"status": "started", "total": 0, "completed": 0}
    
    def stop_batch_processing(self):
        """배치 처리 중단"""
        self.batch_stop_flag = True
        return "⏹️ Stopping batch processing..."
    
    def _batch_worker(self, input_dir, prompt_file, direction, expand_ratio, model_choice, 
                     batch_size, base_seed, gpu_selection, custom_gpu_ids):
        """실제 배치 처리 워커 - GPU 설정 포함"""
        try:
            # 🆕 GPU 설정 파싱
            gpu_mode, gpu_id = self._parse_gpu_selection(gpu_selection)
            selected_gpu_ids = None
            if gpu_mode == "multi" and custom_gpu_ids:
                selected_gpu_ids = [int(gpu.split()[1]) for gpu in custom_gpu_ids]
            exec_config = self._get_execution_config(gpu_mode, gpu_id, selected_gpu_ids)
            
            print(f"🔧 Batch processing with GPU config: {gpu_selection}")
            print(f"🔧 Execution config: nproc={exec_config['nproc_per_node']}, torchrun={exec_config['use_torchrun']}")
            
            # 모델 선택을 모델명으로 변환
            model_name = self._map_model_name(model_choice)
            
            # 베이스 시드 처리
            try:
                base_seed = int(base_seed) if base_seed is not None else 2025
            except:
                base_seed = 2025
            
            # 해상도 기본값 처리
            if batch_size is None:
                batch_size = "480p" if model_name == "vace-1.3B" else "720p"
            ckpt_dir_map = {
                "vace-14B": "models/Wan2.1-VACE-14B",
                "vace-1.3B": "models/Wan2.1-VACE-1.3B"
            }
            ckpt_dir = ckpt_dir_map.get(model_name, "models/Wan2.1-VACE-14B")
            
            # 프롬프트 파일 읽기
            prompts = {}
            if prompt_file:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if '|' in line:
                            filename, prompt = line.strip().split('|', 1)
                            prompts[filename.strip()] = prompt.strip()
            
            # 비디오 파일 찾기
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
            video_files = []
            
            if os.path.exists(input_dir):
                for file in os.listdir(input_dir):
                    if any(file.lower().endswith(ext) for ext in video_extensions):
                        video_files.append(os.path.join(input_dir, file))
            
            total_files = len(video_files)
            completed = 0
            
            for video_file in video_files:
                if self.batch_stop_flag:
                    break
                
                filename = os.path.basename(video_file)
                prompt = prompts.get(filename, "high quality video")
                
                # 🆕 파일별 고유 시드 생성 (베이스 시드 + 인덱스)
                current_seed = base_seed + completed
                
                print(f"🎬 Processing {filename} with {model_choice} model (seed: {current_seed})")
                
                # 🆕 GPU 설정에 따른 명령어 구성
                base_cmd = [
                    'vace/vace_pipeline.py',
                    '--base', 'wan',
                    '--task', 'outpainting',
                    '--video', video_file,
                    '--direction', ','.join(direction) if direction else 'left,right',
                    '--expand_ratio', str(expand_ratio),
                    '--prompt', prompt,
                    '--base_seed', str(current_seed),
                    '--model_name', model_name,
                    '--ckpt_dir', ckpt_dir,
                    '--size', str(batch_size),
                    '--sample_steps', '50',
                    '--sample_guide_scale', '5.0'
                ]
                
                if exec_config['use_torchrun']:
                    cmd = [
                        'torchrun', 
                        f'--nproc-per-node={exec_config["nproc_per_node"]}',
                        '--master_port=12355'
                    ] + base_cmd
                    
                    # 다중 GPU 최적화 옵션
                    if exec_config['nproc_per_node'] > 1:
                        cmd.extend([
                            '--dit_fsdp',
                            '--t5_fsdp',
                            '--ulysses_size', str(exec_config['nproc_per_node']),
                            '--ring_size', '1'
                        ])
                else:
                    cmd = ['python'] + base_cmd
                
                try:
                    # 환경변수 적용
                    batch_env = os.environ.copy()
                    batch_env.update(exec_config['env'])
                    
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        cwd="/data/VACE",
                        env=batch_env
                    )
                    
                    if result.returncode == 0:
                        completed += 1
                        print(f"✅ Completed {filename} ({completed}/{total_files}) with seed {current_seed}")
                    else:
                        print(f"❌ Error processing {filename}: {result.stderr}")
                    
                except Exception as e:
                    print(f"❌ Error processing {filename}: {str(e)}")
                
                time.sleep(1)  # GPU 메모리 정리 대기
            
            print(f"🎉 Batch processing completed! {completed}/{total_files} files processed.")
            
        except Exception as e:
            print(f"❌ Batch worker error: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    # 캡셔닝 기능 상태 출력
    if CAPTIONING_AVAILABLE:
        print("✅ Video captioning feature enabled")
    else:
        print("⚠️ Video captioning feature disabled")
        print("💡 To enable: pip install transformers torch torchvision")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_port', type=int, default=7860)
    parser.add_argument('--server_name', default='0.0.0.0')
    parser.add_argument('--save_dir', default='cache')
    parser.add_argument('--model_name', default='vace-14B')
    parser.add_argument('--ckpt_dir', default='models/Wan2.1-VACE-14B')
    
    args = parser.parse_args()
    
    # 캐시 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 통합 데모 실행
    unified_demo = UnifiedVACEDemo(args)
    demo = unified_demo.create_ui()
    
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=True,
        show_error=True
    )