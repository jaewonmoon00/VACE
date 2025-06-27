# /data/VACE/vace/gradios/unified_vace_demo.py

import gradio as gr
import os
import sys
import time
import threading
import random
import json
import subprocess
import shutil
import logging
from pathlib import Path
import torch

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
        
        # 🆕 배치 상태 관리
        self.batch_status_file = None
        self.batch_timer_running = False
        
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
                
                # Tab 2: 개선된 배치 처리
                with gr.TabItem("📦 Enhanced Batch Processing"):
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
        """🆕 개선된 배치 처리 UI - 실시간 업데이트 포함"""
        gr.Markdown("### 📦 Enhanced Batch Video Processing")
        gr.Markdown("Upload multiple videos directly and see real-time progress.")
        
        with gr.Row():
            with gr.Column():
                # 파일 업로드 섹션
                with gr.Accordion("📁 File Upload", open=True):
                    self.batch_files = gr.File(
                        label="Upload Videos",
                        file_count="multiple",
                        file_types=[".mp4", ".avi", ".mov", ".mkv", ".wmv"],
                        elem_id="batch_file_upload"
                    )
                    
                    with gr.Row():
                        self.batch_load_files_btn = gr.Button("📋 Load Files", variant="secondary")
                        self.batch_clear_files_btn = gr.Button("🗑️ Clear All", variant="secondary")
                
                # GPU 설정
                self.batch_gpu_selection, self.batch_custom_gpu_ids = self.create_gpu_settings_ui("batch")
                
                # 글로벌 설정
                with gr.Accordion("🌐 Global Settings", open=True):
                    with gr.Row():
                        self.batch_direction = gr.CheckboxGroup(
                            choices=["left", "right", "up", "down"],
                            value=["left", "right"],
                            label="Extension Direction",
                            scale=2
                        )
                        self.batch_expand_ratio = gr.Slider(
                            minimum=0.1, maximum=2.0, value=1.6, step=0.1,
                            label="Extension Ratio",
                            scale=1
                        )
                    
                    with gr.Row():
                        self.batch_model = gr.Dropdown(
                            choices=["14B", "1.3B"],
                            value="14B",
                            label="Model",
                            scale=1
                        )
                        self.batch_size = gr.Dropdown(
                            choices=["720p", "480p"],
                            value="720p",
                            label="Output Resolution",
                            scale=1
                        )
                        # 시드 설정
                        with gr.Column(scale=1):
                            with gr.Row():
                                self.batch_base_seed = gr.Number(
                                    value=2025,
                                    label="Base Seed",
                                    scale=4,
                                    info="Each video uses base_seed + index"
                                )
                                self.batch_random_seed_btn = gr.Button(
                                    "🎲", 
                                    variant="secondary",
                                    scale=1,
                                    size="sm"
                                )
                
                # 배치 컨트롤
                with gr.Accordion("🎮 Batch Control", open=True):
                    with gr.Row():
                        self.batch_start_btn = gr.Button("▶️ Start Batch", variant="primary", scale=2)
                        self.batch_pause_btn = gr.Button("⏸️ Pause", variant="secondary", scale=1)
                        self.batch_stop_btn = gr.Button("⏹️ Stop", variant="stop", scale=1)
                    
                    # 전체 진행 상황
                    self.batch_overall_progress = gr.Markdown(
                        value="**Overall Progress:** Ready to start\n\n**Status:** Upload videos to begin",
                        elem_id="batch_overall_progress"
                    )
            
            with gr.Column():
                # 파일별 상세 관리 패널
                with gr.Accordion("📝 File Management", open=True):
                    self.batch_files_container = gr.HTML(
                        value="<div style='text-align: center; padding: 20px; color: #666;'>Upload videos to see file list</div>",
                        elem_id="batch_files_container"
                    )
                    
                    # 숨겨진 상태 저장소
                    self.batch_files_state = gr.State(value=[])
                    self.batch_prompts_state = gr.State(value={})
                    
                    # 전체 캡션 생성 버튼
                    if CAPTIONING_AVAILABLE:
                        with gr.Row():
                            self.batch_caption_all_btn = gr.Button(
                                "🎬 Generate All Captions", 
                                variant="secondary"
                            )
                            self.batch_caption_method = gr.Dropdown(
                                choices=["BLIP-2 (Local)", "GPT-4V (API)"],
                                value="BLIP-2 (Local)",
                                label="Caption Method"
                            )
                
                # 🆕 실시간 처리 현황 (자동 업데이트)
                with gr.Accordion("📊 Live Processing Status", open=True):
                    self.batch_status_display = gr.HTML(
                        value=self._create_empty_status_display(),
                        elem_id="batch_status_display"
                    )
                    
                    # 🆕 결과 갤러리 (실시간 업데이트)
                    self.batch_results_gallery = gr.Gallery(
                        label="✅ Completed Videos",
                        columns=2,
                        rows=2,
                        elem_id="batch_results_gallery",
                        show_label=True,
                        allow_preview=True,
                        preview=True
                    )
                    
                    # 🆕 자동 새로고침 제어
                    with gr.Row():
                        gr.Markdown("**Manual Refresh Only:** Click the refresh button below to update status")
                        self.batch_manual_refresh_btn = gr.Button(
                            "🔄 Refresh Status",
                            variant="secondary",
                            size="sm"
                        )
        
        # 🆕 타이머 컴포넌트 (숨겨진 상태)
        self.batch_timer_state = gr.State(value=False)
        
        # 🆕 실시간 업데이트를 위한 주기적 새로고침 버튼 (숨겨진)
        self.batch_auto_refresh_trigger = gr.Button("Hidden Auto Refresh", visible=False)
        
        # 자동 새로고침을 위한 JavaScript 기반 타이머 설정
        refresh_js = """
        function() {
            if (document.getElementById('batch_auto_refresh').checked) {
                setTimeout(() => {
                    document.getElementById('batch_manual_refresh_btn').click();
                }, 3000);
            }
            return null;
        }
        """
        
        # 이벤트 핸들러 설정
        self._setup_enhanced_batch_events()

    def _create_empty_status_display(self):
        """빈 상태 표시 HTML 생성"""
        return """
        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; background-color: #f9f9f9;">
            <h4 style="margin-top: 0; color: #666;">📊 Processing Queue</h4>
            <p style="color: #999; text-align: center; margin: 20px 0;">
                No files loaded yet.<br>
                Upload videos to see processing status.
            </p>
        </div>
        """

    def _create_file_management_html(self, files_info, prompts_dict):
        """파일 관리 HTML 생성"""
        if not files_info:
            return "<div style='text-align: center; padding: 20px; color: #666;'>Upload videos to see file list</div>"
        
        html_parts = ["<div style='space-y: 10px;'>"]
        
        for idx, file_info in enumerate(files_info):
            filename = file_info.get('name', f'File_{idx}')
            file_size = file_info.get('size', 0)
            size_mb = file_size / (1024 * 1024) if file_size > 0 else 0
            current_prompt = prompts_dict.get(filename, "")
            
            # 각 파일별 카드
            html_parts.append(f"""
            <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 10px; background: white;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <h4 style="margin: 0; color: #333;">📹 {filename}</h4>
                    <span style="color: #666; font-size: 0.9em;">{size_mb:.1f} MB</span>
                </div>
                
                <div style="margin-bottom: 10px;">
                    <label style="display: block; margin-bottom: 5px; font-weight: bold; color: #555;">
                        Description:
                    </label>
                    <textarea 
                        id="prompt_{idx}" 
                        placeholder="Enter description for this video... (or use Auto Caption)"
                        style="width: 100%; height: 60px; padding: 8px; border: 1px solid #ccc; border-radius: 4px; resize: vertical; color:#333;"
                    >{current_prompt}</textarea>
                </div>
                
                <div style="display: flex; gap: 10px;">
                    <button 
                        style="padding: 6px 12px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85em;"
                        {'disabled' if not CAPTIONING_AVAILABLE else ''}
                    >
                        🎬 Auto Caption
                    </button>
                    <button 
                        style="padding: 6px 12px; background: #dc3545; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85em;"
                    >
                        🗑️ Clear
                    </button>
                </div>
            </div>
            """)
        
        html_parts.append("</div>")
        return "".join(html_parts)

    def _setup_enhanced_batch_events(self):
        """🆕 향상된 배치 처리 이벤트 핸들러 설정"""
        
        # 랜덤 시드 버튼
        self.batch_random_seed_btn.click(
            self.randomize_batch_seed,
            outputs=[self.batch_base_seed]
        )
        
        # 파일 로드 버튼
        self.batch_load_files_btn.click(
            self.load_batch_files,
            inputs=[self.batch_files],
            outputs=[
                self.batch_files_container,
                self.batch_files_state,
                self.batch_status_display,
                self.batch_overall_progress
            ]
        )
        
        # 파일 클리어 버튼
        self.batch_clear_files_btn.click(
            self.clear_batch_files,
            outputs=[
                self.batch_files_container,
                self.batch_files_state,
                self.batch_prompts_state,
                self.batch_status_display,
                self.batch_overall_progress,
                self.batch_results_gallery
            ]
        )
        
        # 🆕 향상된 배치 시작 (타이머 포함)
        self.batch_start_btn.click(
            self.start_enhanced_batch_with_monitoring,
            inputs=[
                self.batch_files_state,
                self.batch_prompts_state,
                self.batch_direction,
                self.batch_expand_ratio,
                self.batch_model,
                self.batch_size,
                self.batch_base_seed,
                self.batch_gpu_selection,
                self.batch_custom_gpu_ids
            ],
            outputs=[
                self.batch_overall_progress,
                self.batch_status_display,
                self.batch_timer_state
            ]
        )
        
        # 🆕 배치 중단 (타이머 정지 포함)
        self.batch_stop_btn.click(
            self.stop_enhanced_batch_processing,
            outputs=[
                self.batch_overall_progress,
                self.batch_timer_state
            ]
        )
        
        # 🆕 수동 새로고침
        self.batch_manual_refresh_btn.click(
            self.manual_refresh_batch_status,
            outputs=[
                self.batch_status_display,
                self.batch_results_gallery,
                self.batch_overall_progress
            ]
        )
        
        # 전체 캡션 생성
        if CAPTIONING_AVAILABLE:
            self.batch_caption_all_btn.click(
                self.generate_all_captions,
                inputs=[
                    self.batch_files_state,
                    self.batch_caption_method,
                    self.pipeline_api_key  # API 키 공유
                ],
                outputs=[
                    self.batch_prompts_state,
                    self.batch_files_container,
                    self.batch_overall_progress
                ]
            )
        
        # 모델 변경 시 해상도 업데이트
        self.batch_model.change(
            self.update_batch_size_options,
            inputs=[self.batch_model],
            outputs=[self.batch_size]
        )

    def _create_batch_status_file(self, files_info):
        """🆕 배치 상태 파일 생성"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        status_dir = os.path.join(self.cfg.save_dir, 'batch_status')
        os.makedirs(status_dir, exist_ok=True)
        
        self.batch_status_file = os.path.join(status_dir, f'batch_status_{timestamp}.json')
        
        # 초기 상태
        initial_status = {
            'start_time': time.time(),
            'total_files': len(files_info),
            'completed': 0,
            'failed': 0,
            'current_file': None,
            'processing_status': {},
            'completed_videos': [],
            'error_files': [],
            'is_running': True
        }
        
        # 파일별 상태 초기화
        for file_info in files_info:
            filename = file_info['name']
            initial_status['processing_status'][filename] = {
                'status': 'pending',
                'start_time': None,
                'end_time': None,
                'elapsed': 0,
                'error': None,
                'result_path': None
            }
        
        # 상태 파일 저장
        with open(self.batch_status_file, 'w', encoding='utf-8') as f:
            json.dump(initial_status, f, ensure_ascii=False, indent=2)
        
        return self.batch_status_file

    def _update_batch_status(self, updates):
        """🆕 배치 상태 파일 업데이트"""
        if not self.batch_status_file or not os.path.exists(self.batch_status_file):
            return
        
        try:
            # 현재 상태 읽기
            with open(self.batch_status_file, 'r', encoding='utf-8') as f:
                status = json.load(f)
            
            # 업데이트 적용
            for key, value in updates.items():
                if key == 'processing_status' and isinstance(value, dict):
                    # 중첩된 딕셔너리 업데이트
                    for filename, file_updates in value.items():
                        if filename in status['processing_status']:
                            status['processing_status'][filename].update(file_updates)
                else:
                    status[key] = value
            
            # 상태 파일 다시 저장
            with open(self.batch_status_file, 'w', encoding='utf-8') as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"❌ Failed to update batch status: {e}")

    def _read_batch_status(self):
        """🆕 배치 상태 파일 읽기"""
        if not self.batch_status_file or not os.path.exists(self.batch_status_file):
            return None
        
        try:
            with open(self.batch_status_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Failed to read batch status: {e}")
            return None

    def start_enhanced_batch_with_monitoring(self, files_info, prompts_dict, direction, expand_ratio, 
                                           model_choice, batch_size, base_seed, gpu_selection, custom_gpu_ids):
        """🆕 향상된 배치 처리 시작 (모니터링 포함)"""
        if not files_info:
            return "**Error:** No files loaded for processing", self._create_empty_status_display(), False
        
        if self.batch_thread and self.batch_thread.is_alive():
            return "**Warning:** Batch processing already running", self._create_empty_status_display(), False
        
        # 🆕 상태 파일 생성
        status_file = self._create_batch_status_file(files_info)
        print(f"📄 Created batch status file: {status_file}")
        
        # 타이머 시작
        self.batch_timer_running = True
        self.batch_stop_flag = False
        
        # 백그라운드 스레드 시작
        self.batch_thread = threading.Thread(
            target=self._enhanced_batch_worker_with_status,
            args=(files_info, prompts_dict, direction, expand_ratio, model_choice, 
                  batch_size, base_seed, gpu_selection, custom_gpu_ids)
        )
        self.batch_thread.start()
        
        # 초기 상태 HTML
        initial_status = self._read_batch_status()
        status_html = self._create_status_display_html_from_status(initial_status)
        
        progress_text = f"**Batch Processing Started**\n\n**Total Files:** {len(files_info)}\n**Status:** Initializing...\n**Tip:** Use the 'Refresh Status' button to check progress"
        
        return progress_text, status_html, True

    def _enhanced_batch_worker_with_status(self, files_info, prompts_dict, direction, expand_ratio, 
                                         model_choice, batch_size, base_seed, gpu_selection, custom_gpu_ids):
        """🆕 향상된 배치 처리 워커 (상태 업데이트 포함)"""
        try:
            # GPU 설정 파싱
            gpu_mode, gpu_id = self._parse_gpu_selection(gpu_selection)
            selected_gpu_ids = None
            if gpu_mode == "multi" and custom_gpu_ids:
                selected_gpu_ids = [int(gpu.split()[1]) for gpu in custom_gpu_ids]
            exec_config = self._get_execution_config(gpu_mode, gpu_id, selected_gpu_ids)
            
            model_name = self._map_model_name(model_choice)
            base_seed = int(base_seed) if base_seed is not None else 2025
            
            if batch_size is None:
                batch_size = "480p" if model_name == "vace-1.3B" else "720p"
            
            ckpt_dir_map = {
                "vace-14B": "models/Wan2.1-VACE-14B",
                "vace-1.3B": "models/Wan2.1-VACE-1.3B"
            }
            ckpt_dir = ckpt_dir_map.get(model_name, "models/Wan2.1-VACE-14B")
            
            total_files = len(files_info)
            completed = 0
            failed = 0
            
            for idx, file_info in enumerate(files_info):
                if self.batch_stop_flag:
                    break
                
                filename = file_info['name']
                file_path = file_info['path']
                
                if not os.path.exists(file_path):
                    print(f"❌ File not found: {file_path}")
                    failed += 1
                    self._update_batch_status({
                        'failed': failed,
                        'processing_status': {
                            filename: {
                                'status': 'failed',
                                'error': 'File not found',
                                'end_time': time.time()
                            }
                        }
                    })
                    continue
                
                prompt = prompts_dict.get(filename, "high quality video")
                current_seed = base_seed + idx
                
                print(f"🎬 Processing {filename} ({idx + 1}/{total_files}) with seed {current_seed}")
                
                # 🆕 처리 시작 상태 업데이트
                start_time = time.time()
                self._update_batch_status({
                    'current_file': filename,
                    'processing_status': {
                        filename: {
                            'status': 'processing',
                            'start_time': start_time
                        }
                    }
                })
                
                try:
                    # GPU 설정에 따른 명령어 구성
                    base_cmd = [
                        'vace/vace_pipeline.py',
                        '--base', 'wan',
                        '--task', 'outpainting',
                        '--video', file_path,
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
                        
                        if exec_config['nproc_per_node'] > 1:
                            cmd.extend([
                                '--dit_fsdp',
                                '--t5_fsdp',
                                '--ulysses_size', str(exec_config['nproc_per_node']),
                                '--ring_size', '1'
                            ])
                    else:
                        cmd = ['python'] + base_cmd
                    
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
                    
                    elapsed_time = time.time() - start_time
                    
                    if result.returncode == 0:
                        completed += 1
                        
                        # 🆕 결과 파일 경로 찾기
                        result_path = self._find_result_video(filename, model_name)
                        
                        print(f"✅ Completed {filename} ({completed}/{total_files}) in {elapsed_time:.1f}s")
                        
                        # 🆕 성공 상태 업데이트
                        self._update_batch_status({
                            'completed': completed,
                            'current_file': None,
                            'processing_status': {
                                filename: {
                                    'status': 'completed',
                                    'end_time': time.time(),
                                    'elapsed': elapsed_time,
                                    'result_path': result_path
                                }
                            }
                        })
                        
                        # 🆕 완료된 비디오 목록 업데이트
                        if result_path and os.path.exists(result_path):
                            status = self._read_batch_status()
                            if status:
                                completed_videos = status.get('completed_videos', [])
                                completed_videos.append({
                                    'filename': filename,
                                    'path': result_path,
                                    'completion_time': time.time()
                                })
                                self._update_batch_status({'completed_videos': completed_videos})
                    else:
                        failed += 1
                        error_msg = result.stderr if result.stderr else "Unknown error"
                        print(f"❌ Error processing {filename}: {error_msg}")
                        
                        # 🆕 실패 상태 업데이트
                        self._update_batch_status({
                            'failed': failed,
                            'current_file': None,
                            'processing_status': {
                                filename: {
                                    'status': 'failed',
                                    'end_time': time.time(),
                                    'elapsed': elapsed_time,
                                    'error': error_msg[:200]  # 에러 메시지 길이 제한
                                }
                            }
                        })
                    
                except Exception as e:
                    failed += 1
                    elapsed_time = time.time() - start_time
                    print(f"❌ Error processing {filename}: {str(e)}")
                    
                    # 🆕 예외 상태 업데이트
                    self._update_batch_status({
                        'failed': failed,
                        'current_file': None,
                        'processing_status': {
                            filename: {
                                'status': 'failed',
                                'end_time': time.time(),
                                'elapsed': elapsed_time,
                                'error': str(e)[:200]
                            }
                        }
                    })
                
                time.sleep(1)  # GPU 메모리 정리 대기
            
            # 🆕 최종 상태 업데이트
            self._update_batch_status({
                'is_running': False,
                'current_file': None,
                'end_time': time.time()
            })
            
            print(f"🎉 Enhanced batch processing completed! {completed}/{total_files} files processed.")
            
        except Exception as e:
            print(f"❌ Enhanced batch worker error: {str(e)}")
            self._update_batch_status({
                'is_running': False,
                'error': str(e)
            })
        finally:
            self.batch_timer_running = False

    def _find_result_video(self, filename, model_name):
        """🆕 결과 비디오 파일 경로 찾기"""
        try:
            # 결과 디렉토리 패턴들
            base_name = Path(filename).stem
            results_dir = f"/data/VACE/results/{model_name}"
            
            if not os.path.exists(results_dir):
                return None
            
            # 최근 생성된 디렉토리 찾기 (타임스탬프 기준)
            potential_dirs = []
            for item in os.listdir(results_dir):
                item_path = os.path.join(results_dir, item)
                if os.path.isdir(item_path) and base_name in item:
                    potential_dirs.append((item_path, os.path.getctime(item_path)))
            
            if not potential_dirs:
                return None
            
            # 가장 최근 디렉토리
            latest_dir = max(potential_dirs, key=lambda x: x[1])[0]
            result_video = os.path.join(latest_dir, 'out_video.mp4')
            
            if os.path.exists(result_video):
                return result_video
            
            return None
            
        except Exception as e:
            print(f"❌ Error finding result video for {filename}: {e}")
            return None

    def manual_refresh_batch_status(self):
        """🆕 수동 새로고침"""
        try:
            status = self._read_batch_status()
            if not status:
                empty_status = self._create_empty_status_display()
                empty_progress = "**Overall Progress:** No active batch\n\n**Status:** No status file found"
                return empty_status, [], empty_progress
            
            # 상태 HTML 생성
            status_html = self._create_status_display_html_from_status(status)
            
            # 결과 갤러리 업데이트
            gallery_items = []
            completed_videos = status.get('completed_videos', [])
            for video_info in completed_videos:
                video_path = video_info.get('path')
                if video_path and os.path.exists(video_path):
                    gallery_items.append(video_path)
            
            # 전체 진행률 업데이트
            total = status.get('total_files', 0)
            completed = status.get('completed', 0)
            failed = status.get('failed', 0)
            is_running = status.get('is_running', False)
            
            if total > 0:
                progress_percent = (completed / total) * 100
                status_text = "Running..." if is_running else "Completed"
                progress_text = f"""**Batch Processing Progress**

**Total Files:** {total}
**Completed:** {completed} ({progress_percent:.1f}%)
**Failed:** {failed}
**Status:** {status_text}

**Last Updated:** {time.strftime('%H:%M:%S')}"""
            else:
                progress_text = "**Overall Progress:** No active batch"
            
            return status_html, gallery_items, progress_text
            
        except Exception as e:
            error_html = f"<div style='color: red; padding: 20px;'>Error refreshing status: {str(e)}</div>"
            error_progress = f"**Error:** Failed to refresh - {str(e)}"
            return error_html, [], error_progress

    def _create_status_display_html_from_status(self, status):
        """🆕 상태 객체로부터 HTML 생성"""
        if not status:
            return self._create_empty_status_display()
        
        total = status.get('total_files', 0)
        completed = status.get('completed', 0)
        failed = status.get('failed', 0)
        is_running = status.get('is_running', False)
        current_file = status.get('current_file')
        processing_status = status.get('processing_status', {})
        
        # 상태별 카운트
        pending = 0
        processing = 0
        
        for file_status in processing_status.values():
            if file_status['status'] == 'pending':
                pending += 1
            elif file_status['status'] == 'processing':
                processing += 1
        
        # 진행률 계산
        progress_percent = (completed / total * 100) if total > 0 else 0
        
        # 현재 처리 중인 파일 표시
        current_status = ""
        if is_running and current_file:
            current_status = f"<div style='background: #e3f2fd; padding: 10px; border-radius: 5px; margin-bottom: 15px; color:#333;'><strong>🔄 Currently Processing:</strong> {current_file}</div>"
        elif not is_running:
            current_status = f"<div style='background: #e8f5e8; padding: 10px; border-radius: 5px; margin-bottom: 15px; color:#333'><strong>✅ Batch Processing Complete!</strong></div>"
        
        html = f"""
        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; background-color: #f9f9f9;">
            <h4 style="margin-top: 0; color: #333;">📊 Real-time Processing Status</h4>
            
            {current_status}
            
            <!-- 전체 진행률 -->
            <div style="margin-bottom: 15px; color: #333;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px; color: #333;">
                    <span style="font-weight: bold; color:#333;">Overall Progress</span>
                    <span style="color:#333">{completed}/{total} completed ({progress_percent:.1f}%)</span>
                </div>
                <div style="width: 100%; height: 20px; background-color: #e9ecef; border-radius: 10px; overflow: hidden;">
                    <div style="width: {progress_percent}%; height: 100%; background: linear-gradient(90deg, #28a745, #20c997); transition: width 0.3s ease;"></div>
                </div>
            </div>
            
            <!-- 상태별 카운트 -->
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 15px;">
                <div style="text-align: center; padding: 8px; background: #fff3cd; border-radius: 4px;">
                    <div style="font-weight: bold; color: #856404;">⏳ Pending</div>
                    <div style="font-size: 1.2em; color: #856404;">{pending}</div>
                </div>
                <div style="text-align: center; padding: 8px; background: #cce5ff; border-radius: 4px;">
                    <div style="font-weight: bold; color: #004085;">🔄 Processing</div>
                    <div style="font-size: 1.2em; color: #004085;">{processing}</div>
                </div>
                <div style="text-align: center; padding: 8px; background: #d4edda; border-radius: 4px;">
                    <div style="font-weight: bold; color: #155724;">✅ Completed</div>
                    <div style="font-size: 1.2em; color: #155724;">{completed}</div>
                </div>
                <div style="text-align: center; padding: 8px; background: #f8d7da; border-radius: 4px;">
                    <div style="font-weight: bold; color: #721c24;">❌ Failed</div>
                    <div style="font-size: 1.2em; color: #721c24;">{failed}</div>
                </div>
            </div>
            
            <!-- 개별 파일 상태 -->
            <div style="max-height: 300px; overflow-y: auto;">
        """
        
        # 개별 파일 상태 추가
        for filename, file_status in processing_status.items():
            status_icon = {
                'pending': '⏳',
                'processing': '🔄',
                'completed': '✅',
                'failed': '❌'
            }.get(file_status['status'], '❓')
            
            status_color = {
                'pending': '#856404',
                'processing': '#004085',
                'completed': '#155724',
                'failed': '#721c24'
            }.get(file_status['status'], '#333')
            
            elapsed_time = file_status.get('elapsed', 0)
            time_str = f"{elapsed_time:.1f}s" if elapsed_time > 0 else "-"
            
            # 에러 메시지 표시
            error_info = ""
            if file_status['status'] == 'failed' and file_status.get('error'):
                error_info = f"<div style='font-size: 0.8em; color: #dc3545; margin-top: 2px;'>Error: {file_status['error'][:50]}...</div>"
            
            html += f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; margin-bottom: 5px; background: white; border-radius: 4px; border-left: 4px solid {status_color};">
                <div style="flex: 1;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 1.2em;">{status_icon}</span>
                        <span style="font-weight: 500; color:#333;">{filename}</span>
                    </div>
                    {error_info}
                </div>
                <div style="text-align: right; font-size: 0.9em; color: #333;">
                    <div style="color: {status_color}; font-weight: bold;">{file_status['status'].title()}</div>
                    <div style="color: #333;">Time: {time_str}</div>
                </div>
            </div>
            """
        
        html += "</div></div>"
        return html

    def stop_enhanced_batch_processing(self):
        """🆕 향상된 배치 처리 중단"""
        self.batch_stop_flag = True
        self.batch_timer_running = False
        
        # 상태 파일 업데이트
        if self.batch_status_file:
            self._update_batch_status({
                'is_running': False,
                'current_file': None,
                'stopped_by_user': True
            })
        
        return "⏹️ Batch processing stopped by user", False

    def load_batch_files(self, uploaded_files):
        """업로드된 파일들 로드 및 정보 추출"""
        if not uploaded_files:
            empty_html = "<div style='text-align: center; padding: 20px; color: #666;'>Upload videos to see file list</div>"
            empty_status = self._create_empty_status_display()
            empty_progress = "**Overall Progress:** No files loaded\n\n**Status:** Upload videos to begin"
            return empty_html, [], empty_status, empty_progress
        
        files_info = []
        processing_status = {}
        
        for file_path in uploaded_files:
            try:
                filename = os.path.basename(file_path)
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                
                file_info = {
                    'name': filename,
                    'path': file_path,
                    'size': file_size
                }
                files_info.append(file_info)
                
                # 초기 처리 상태 설정
                processing_status[filename] = {
                    'status': 'pending',
                    'elapsed': 0,
                    'error': None
                }
                
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
        
        # HTML 생성
        files_html = self._create_file_management_html(files_info, {})
        status_html = self._create_status_display_html_from_status({'processing_status': processing_status, 'total_files': len(files_info), 'completed': 0, 'failed': 0, 'is_running': False})
        progress_text = f"**Overall Progress:** {len(files_info)} files loaded\n\n**Status:** Ready to start batch processing"
        
        return files_html, files_info, status_html, progress_text

    def clear_batch_files(self):
        """모든 파일 및 상태 클리어"""
        empty_html = "<div style='text-align: center; padding: 20px; color: #666;'>Upload videos to see file list</div>"
        empty_status = self._create_empty_status_display()
        empty_progress = "**Overall Progress:** Cleared\n\n**Status:** Upload videos to begin"
        
        return empty_html, [], {}, empty_status, empty_progress, []

    def update_batch_size_options(self, model_choice):
        """모델 변경 시 해상도 옵션 업데이트"""
        if model_choice == "14B":
            return gr.update(choices=["720p", "480p", "720×1280", "1280×720"], value="720p")
        elif model_choice == "1.3B":
            return gr.update(choices=["480p", "480×832", "832×480"], value="480p")
        return gr.update(choices=["720p", "480p"], value="720p")

    def generate_all_captions(self, files_info, caption_method, api_key):
        """모든 파일에 대해 캡션 생성"""
        if not files_info:
            return {}, "<div style='text-align: center; padding: 20px; color: #666;'>No files loaded</div>", "**Status:** No files to caption"
        
        updated_prompts = {}
        
        try:
            total_files = len(files_info)
            
            for idx, file_info in enumerate(files_info):
                filename = file_info['name']
                file_path = file_info['path']
                
                if not os.path.exists(file_path):
                    continue
                    
                progress_msg = f"**Captioning Progress:** {idx + 1}/{total_files}\n\n**Current:** {filename}"
                
                # 캡션 생성
                caption, status = self.generate_auto_caption(file_path, caption_method, api_key)
                
                if caption and not caption.startswith(("Error:", "Failed", "❌")):
                    updated_prompts[filename] = caption
            
            # 업데이트된 HTML 생성
            updated_html = self._create_file_management_html(files_info, updated_prompts)
            final_progress = f"**Captioning Complete:** {len(updated_prompts)}/{total_files} successful\n\n**Status:** Ready for batch processing"
            
            return updated_prompts, updated_html, final_progress
            
        except Exception as e:
            error_html = f"<div style='color: red; padding: 20px;'>Error generating captions: {str(e)}</div>"
            error_progress = f"**Error:** Caption generation failed - {str(e)}"
            return {}, error_html, error_progress

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