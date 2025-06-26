import gradio as gr
from gradio import themes
import os
import sys
import time
import threading
from pathlib import Path
import subprocess
import torch

# 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from gradios.vace_preprocess_demo import VACETag
    from gradios.vace_wan_demo import VACEInference
except ImportError:
    # 대체 경로
    from vace_preprocess_demo import VACETag
    from vace_wan_demo import VACEInference

class UnifiedVACEDemo:
    def __init__(self, cfg):
        self.cfg = cfg
        self.batch_stop_flag = False
        self.batch_thread = None
        
        # 각 모듈 초기화 (지연 로딩)
        self.preprocessor = None
        self.inference = None
        
    def init_preprocessor(self):
        """전처리 모듈 지연 초기화"""
        if self.preprocessor is None:
            self.preprocessor = VACETag(self.cfg)
        return self.preprocessor
    
    def init_inference(self):
        """추론 모듈 지연 초기화"""
        if self.inference is None:
            self.inference = VACEInference(self.cfg, skip_load=True)
        return self.inference
    def create_ui(self):
        with gr.Blocks(title="VACE Unified Demo", theme=themes.Soft()) as demo:
            gr.Markdown("""
            <div style="text-align: center; font-size: 28px; font-weight: bold; margin-bottom: 20px;">
                🎬 Freewillusion Video Extender
            </div>
            <div style="text-align: center; font-size: 16px; color: #666; margin-bottom: 20px;">
                Complete pipeline for video preprocessing, inference, and batch processing
            </div>
            """)
            
            with gr.Tabs() as main_tabs:
                # Tab 1: 전처리
                with gr.TabItem("📋 Preprocessing", id="preprocess"):
                    with gr.Row():
                        gr.Markdown("### Video Preprocessing Module")
                        init_preprocess_btn = gr.Button("🔄 Initialize Preprocessor", size="sm")
                    
                    preprocess_container = gr.HTML("<p>Click 'Initialize Preprocessor' to load the preprocessing interface.</p>")
                    
                    def load_preprocessor():
                        preprocessor = self.init_preprocessor()
                        # 실제로는 preprocessor.create_ui() 결과를 반환해야 하지만
                        # Gradio 동적 UI 생성의 한계로 인해 메시지만 표시
                        return "<p>✅ Preprocessor loaded! Please use the individual demo files for full functionality.</p>"
                    
                    init_preprocess_btn.click(load_preprocessor, outputs=[preprocess_container])
                
                # Tab 2: 추론
                with gr.TabItem("🚀 Inference", id="inference"):
                    with gr.Row():
                        gr.Markdown("### Video Inference Module")
                        init_inference_btn = gr.Button("🔄 Initialize Inference", size="sm")
                    
                    inference_container = gr.HTML("<p>Click 'Initialize Inference' to load the inference interface.</p>")
                    
                    def load_inference():
                        inference = self.init_inference()
                        return "<p>✅ Inference engine loaded! Please use the individual demo files for full functionality.</p>"
                    
                    init_inference_btn.click(load_inference, outputs=[inference_container])
                
                # Tab 3: 통합 파이프라인
                with gr.TabItem("⚡ Full Pipeline", id="pipeline"):
                    self.create_pipeline_ui()
                
                # Tab 4: 배치 처리
                with gr.TabItem("📦 Batch Processing", id="batch"):
                    self.create_batch_ui()
            
        return demo
    
    def create_pipeline_ui(self):
        """전처리 + 추론을 한 번에 실행하는 UI"""
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        gpu_info = f"🖥️ Available GPUs: {gpu_count}" if gpu_count > 0 else "⚠️ No GPU detected"
        
        gr.Markdown(f"### 🔄 End-to-End Pipeline")
        gr.Markdown(f"Upload a video and configure settings to run the complete preprocessing + inference pipeline.")
        gr.Markdown(f"**{gpu_info}** - Will use up to {min(gpu_count, 2)} GPUs for parallel processing")
    
        with gr.Row():
            with gr.Column(scale=1):
                # ✅ 모델 선택 추가
                self.pipeline_model = gr.Dropdown(
                    choices=["vace-14B", "vace-1.3B"],
                    value="vace-14B",
                    label="🤖 Model Selection",
                    info="14B: Higher quality, more VRAM | 1.3B: Faster, less VRAM"
                )

                # 입력
                self.pipeline_video = gr.Video(label="Input Video", height=300)
                self.pipeline_task = gr.Dropdown(
                    choices=["outpainting"],
                    value="outpainting",
                    label="Task Type",
                    interactive=False,
                    info="Currently supports outpainting only"
                )

                # ✅ 해상도 선택 (모델별 지원 해상도)
                self.pipeline_size = gr.Dropdown(
                    choices=["720p", "480p"],  # 기본값, 모델 변경 시 업데이트됨
                    value="720p",
                    label="📐 Output Resolution"
                )
                
                # 전처리 설정
                with gr.Accordion("Preprocessing Settings", open=True):
                    self.pipeline_direction = gr.CheckboxGroup(
                        choices=["left", "right", "up", "down"],
                        value=["left", "right"],
                        label="Outpainting Direction"
                    )
                    self.pipeline_expand_ratio = gr.Slider(
                        minimum=0.1, maximum=2.0, value=1.6, step=0.1,
                        label="Expand Ratio"
                    )
                
                # 추론 설정
                with gr.Accordion("Inference Settings", open=True):
                    self.pipeline_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe what you want to generate...",
                        lines=3
                    )
                    self.pipeline_use_prompt_extend = gr.Dropdown(
                        choices=["plain", "wan_zh", "wan_en"],
                        value="plain",
                        label="Prompt Extension"
                    )
                    self.pipeline_seed = gr.Number(value=2025, label="Seed", precision=0)

                    # ✅ 고급 설정
                    with gr.Accordion("Advanced Settings", open=False):
                        self.pipeline_sampling_steps = gr.Slider(
                            minimum=20, maximum=100, value=50, step=5,
                            label="Sampling Steps"
                        )
                        self.pipeline_guide_scale = gr.Slider(
                            minimum=1.0, maximum=10.0, value=5.0, step=0.5,
                            label="Guidance Scale"
                        )


                self.pipeline_run_btn = gr.Button("🚀 Run Full Pipeline", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                # 결과 표시
                self.pipeline_progress = gr.Textbox(
                    label="Progress", 
                    interactive=False,
                    max_lines=5,
                    placeholder="Ready to process..."
                )
                self.pipeline_result_video = gr.Video(label="Result Video", height=300)
                self.pipeline_intermediate = gr.Gallery(
                    label="Intermediate Results",
                    columns=2, rows=1,
                    height=200
                )

                # ✅ 모델 정보 표시
                self.pipeline_model_info = gr.HTML(
                    value=self._get_model_info("vace-14B"),
                    label="Model Information"
                )
        
        # ✅ 모델 변경 시 해상도 옵션 업데이트
        def update_size_options(model_name):
            from models.wan.configs import SUPPORTED_SIZES
            
            if model_name in SUPPORTED_SIZES:
                supported_sizes = SUPPORTED_SIZES[model_name]
                # 기본값 설정
                default_size = "720p" if "720p" in supported_sizes else supported_sizes[0]
                return gr.Dropdown(choices=list(supported_sizes), value=default_size), self._get_model_info(model_name)
            return gr.Dropdown(choices=["720p", "480p"], value="720p"), self._get_model_info(model_name)
        
        # 모델 변경 시 콜백
        self.pipeline_model.change(
            update_size_options,
            inputs=[self.pipeline_model],
            outputs=[self.pipeline_size, self.pipeline_model_info]
        )

        # 콜백 설정
        self.pipeline_run_btn.click(
            self.run_full_pipeline,
            inputs=[
                self.pipeline_video, self.pipeline_task, 
                self.pipeline_direction, self.pipeline_expand_ratio,
                self.pipeline_prompt, self.pipeline_use_prompt_extend,
                self.pipeline_seed, self.pipeline_model, self.pipeline_size,  # ✅ 누락된 파라미터 추가
                self.pipeline_sampling_steps, self.pipeline_guide_scale      # ✅ 누락된 파라미터 추가
            ],
            outputs=[
                self.pipeline_progress, self.pipeline_result_video,
                self.pipeline_intermediate
            ]
        )
    
    def _get_model_info(self, model_name):
        """모델 정보 HTML 생성"""
        if model_name == "vace-14B":
            return """
            <div style="background: #f0f8ff; padding: 10px; border-radius: 5px; margin: 10px 0;">
                <h4>🚀 VACE-14B Model</h4>
                <ul>
                    <li><strong>Parameters:</strong> 14 Billion</li>
                    <li><strong>VRAM:</strong> ~24GB+ recommended</li>
                    <li><strong>Quality:</strong> Highest</li>
                    <li><strong>Resolutions:</strong> 720p, 480p, 720×1280, 1280×720</li>
                    <li><strong>Speed:</strong> Slower but better quality</li>
                </ul>
            </div>
            """
        elif model_name == "vace-1.3B":
            return """
            <div style="background: #f0fff0; padding: 10px; border-radius: 5px; margin: 10px 0;">
                <h4>⚡ VACE-1.3B Model</h4>
                <ul>
                    <li><strong>Parameters:</strong> 1.3 Billion</li>
                    <li><strong>VRAM:</strong> ~12GB+ recommended</li>
                    <li><strong>Quality:</strong> Good</li>
                    <li><strong>Resolutions:</strong> 480p, 480×832, 832×480</li>
                    <li><strong>Speed:</strong> Faster processing</li>
                </ul>
            </div>
            """
        return ""

    def create_batch_ui(self):
        """배치 처리를 위한 UI - 모델 선택 추가"""
        gr.Markdown("### 📦 Batch Video Processing")
        gr.Markdown("Process multiple videos with different prompts automatically.")
        
        with gr.Row():
            with gr.Column():
                # ✅ 배치용 모델 선택
                self.batch_model = gr.Dropdown(
                    choices=["vace-14B", "vace-1.3B"],
                    value="vace-14B",
                    label="🤖 Batch Model Selection"
                )
                
                self.batch_input_dir = gr.Textbox(
                    value="inputs/",
                    label="Input Directory",
                    placeholder="Path to directory containing videos"
                )
                self.batch_prompt_file = gr.File(
                    label="Prompt File",
                    file_types=[".txt"],
                    type="filepath"
                )
                
                # 배치 설정
                with gr.Accordion("Batch Settings", open=True):
                    self.batch_task = gr.Dropdown(
                        choices=["outpainting"],
                        value="outpainting",
                        label="Task",
                        interactive=False,
                        info="Currently supports outpainting only"
                    )
                    self.batch_direction = gr.CheckboxGroup(
                        choices=["left", "right", "up", "down"],
                        value=["left", "right"],
                        label="Direction"
                    )
                    self.batch_expand_ratio = gr.Slider(
                        minimum=0.1, maximum=2.0, value=1.6,
                        label="Expand Ratio"
                    )
                
                with gr.Row():
                    self.batch_start_btn = gr.Button("▶️ Start Batch", variant="primary")
                    self.batch_stop_btn = gr.Button("⏹️ Stop", variant="stop")
                
                self.batch_progress = gr.Textbox(
                    label="Batch Progress",
                    max_lines=10,
                    interactive=False,
                    placeholder="Ready to start batch processing..."
                )
            
            with gr.Column():
                self.batch_results = gr.Gallery(
                    label="Batch Results",
                    columns=2, rows=3,
                    height=400
                )
                self.batch_status = gr.JSON(
                    label="Processing Status",
                    value={"total": 0, "completed": 0, "failed": 0, "current": ""}
                )
        
        # 배치 처리 콜백
        self.batch_start_btn.click(
            self.start_batch_processing,
            inputs=[
                self.batch_input_dir, self.batch_prompt_file,
                self.batch_task, self.batch_direction, self.batch_expand_ratio
            ],
            outputs=[self.batch_progress, self.batch_status]
        )
        
        self.batch_stop_btn.click(
            self.stop_batch_processing,
            outputs=[self.batch_progress]
        )

    def run_full_pipeline(self, video, task, direction, expand_ratio, prompt, use_prompt_extend, 
             seed, model_name, size, sampling_steps, guide_scale):
        """전체 파이프라인 실행 - 전처리 후 추론 분리"""
        import shutil
        import time
        import subprocess
        from pathlib import Path

        try:
            if not video:
                yield "❌ Please upload a video first!", None, []
                return

            # 파라미터 기본값 처리
            if model_name is None:
                model_name = "vace-14B"
            if size is None:
                size = "720p"
            if sampling_steps is None:
                sampling_steps = 50
            if guide_scale is None:
                guide_scale = 5.0
            if seed is None:
                seed = 2025

            try:
                seed = int(seed)
                sampling_steps = int(sampling_steps)
                guide_scale = float(guide_scale)
                expand_ratio = float(expand_ratio)
            except Exception as e:
                yield f"❌ Parameter conversion error: {str(e)}", None, []
                return

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
                yield f"❌ File copy failed: {e}", None, []
                return

            # 타임스탬프 및 디렉토리
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

            # GPU 설정
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
            nproc_per_node = min(gpu_count, 2)

            # 1. Preprocess 단계
            preprocess_cmd = [
                'python', 'vace/vace_preproccess.py',
                '--task', str(task),
                '--direction', ','.join(direction) if direction else 'left,right',
                '--expand_ratio', str(expand_ratio),
                '--video', str(target_video_path),
                '--pre_save_dir', pre_save_dir
            ]
            yield "🚀 Running preprocessing...", None, []
            try:
                result = subprocess.run(
                    preprocess_cmd,
                    capture_output=True,
                    text=True,
                    cwd=vace_root_dir
                )
                if result.returncode != 0:
                    error_msg = result.stderr if result.stderr else result.stdout
                    yield f"❌ Preprocessing failed:\n{error_msg}", None, []
                    return
            except Exception as e:
                yield f"❌ Preprocessing execution error: {str(e)}", None, []
                return

            # 2. Inference 단계
            src_video_path = os.path.join(pre_save_dir, f"src_video-{task}.mp4")
            src_mask_path = os.path.join(pre_save_dir, f"src_mask-{task}.mp4")

            print(src_video_path, src_mask_path)
            inference_cmd = [
                'torchrun',
                f'--nproc-per-node={nproc_per_node}',
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
            if nproc_per_node > 1:
                inference_cmd.extend([
                    '--dit_fsdp',
                    '--t5_fsdp',
                    '--ulysses_size', str(nproc_per_node),
                    '--ring_size', '1'
                ])
            if prompt and prompt.strip():
                inference_cmd.extend(['--prompt', str(prompt)])
            else:
                inference_cmd.extend(['--prompt', ""])
            if use_prompt_extend and use_prompt_extend != 'plain':
                inference_cmd.extend(['--use_prompt_extend', str(use_prompt_extend)])

            yield "🚀 Running inference...", None, []
            try:
                result = subprocess.run(
                    inference_cmd,
                    capture_output=True,
                    text=True,
                    cwd=vace_root_dir
                )
                if result.returncode == 0:
                    out_video_path = os.path.join(result_save_dir, 'out_video.mp4')
                    intermediates = []
                    if os.path.exists(out_video_path):
                        yield "✅ Pipeline completed!", out_video_path, intermediates
                    else:
                        yield "⚠️ Pipeline finished but result not found.", None, []
                else:
                    error_msg = result.stderr if result.stderr else result.stdout
                    yield f"❌ Inference failed:\n{error_msg}", None, []
            except Exception as e:
                yield f"❌ Inference execution error: {str(e)}", None, []
        except Exception as e:
            yield f"❌ Error: {str(e)}", None, []

    def _find_pipeline_results(self, vace_root_dir, original_name, task, execution_start_time):
        """파이프라인 결과 파일을 정확히 찾기 - 단일 타임스탬프 패턴"""
        result_video = None
        intermediate_files = []
        
        try:
            # ✅ 1. results 디렉토리에서 최신 결과 찾기
            results_dir = os.path.join(vace_root_dir, 'results')
            if os.path.exists(results_dir):
                model_dirs = []
                for model_name in os.listdir(results_dir):
                    model_path = os.path.join(results_dir, model_name)
                    if os.path.isdir(model_path):
                        for result_folder in os.listdir(model_path):
                            result_path = os.path.join(model_path, result_folder)
                            if os.path.isdir(result_path):
                                folder_mtime = os.path.getmtime(result_path)
                                if folder_mtime >= execution_start_time:
                                    model_dirs.append(result_path)
                
                if model_dirs:
                    latest_result_dir = max(model_dirs, key=os.path.getmtime)
                    out_video_path = os.path.join(latest_result_dir, 'out_video.mp4')
                    if os.path.exists(out_video_path):
                        result_video = out_video_path
            
            # ✅ 2. processed 디렉토리에서 중간 결과 찾기 (수정된 패턴)
            processed_dir = os.path.join(vace_root_dir, 'processed')
            if os.path.exists(processed_dir):
                task_dir = os.path.join(processed_dir, task)
                if os.path.exists(task_dir):
                    import re
                    
                    # ✅ 단일 타임스탬프 패턴: {파일명}_{YYYY-MM-DD-HH-MM-SS}
                    timestamp_pattern = r'_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$'
                    
                    matching_folders = []
                    for folder_name in os.listdir(task_dir):
                        folder_path = os.path.join(task_dir, folder_name)
                        if os.path.isdir(folder_path):
                            folder_mtime = os.path.getmtime(folder_path)
                            
                            if folder_mtime >= execution_start_time:
                                # 타임스탬프 제거하여 원본 파일명 추출
                                clean_folder_name = re.sub(timestamp_pattern, '', folder_name)
                                
                                print(f"🔍 Checking: {folder_name}")
                                print(f"   Clean: {clean_folder_name}")
                                print(f"   Original: {original_name}")
                                
                                # ✅ 정확한 매칭 또는 부분 매칭
                                if (clean_folder_name == original_name or
                                    clean_folder_name in original_name or 
                                    original_name in clean_folder_name):
                                    
                                    matching_folders.append((folder_path, folder_mtime))
                                    print(f"✅ 매칭 성공: {folder_name}")
                    
                    if matching_folders:
                        latest_processed_dir = max(matching_folders, key=lambda x: x[1])[0]
                        print(f"📁 선택된 폴더: {os.path.basename(latest_processed_dir)}")
                        
                        # src_video와 src_mask 찾기
                        src_video = os.path.join(latest_processed_dir, f'src_video-{task}.mp4')
                        src_mask = os.path.join(latest_processed_dir, f'src_mask-{task}.mp4')
                        
                        if os.path.exists(src_video):
                            intermediate_files.append(src_video)
                            print(f"✅ Found src_video: {src_video}")
                        if os.path.exists(src_mask):
                            intermediate_files.append(src_mask)
                            print(f"✅ Found src_mask: {src_mask}")
                    else:
                        print(f"❌ No matching folders found")
                        print(f"Available folders:")
                        for folder_name in os.listdir(task_dir):
                            print(f"  - {folder_name}")
            
            return result_video, intermediate_files[:2]
            
        except Exception as e:
            print(f"Error finding results: {e}")
            return None, []
    
    def start_batch_processing(self, input_dir, prompt_file, task, direction, expand_ratio):
        """배치 처리 시작"""
        if self.batch_thread and self.batch_thread.is_alive():
            return "⚠️ Batch processing is already running!", {"status": "running"}
        
        self.batch_stop_flag = False
        self.batch_thread = threading.Thread(
            target=self._batch_worker,
            args=(input_dir, prompt_file, task, direction, expand_ratio)
        )
        self.batch_thread.start()
        
        return "🚀 Batch processing started!", {"status": "started", "total": 0, "completed": 0}
    
    def stop_batch_processing(self):
        """배치 처리 중단"""
        self.batch_stop_flag = True
        return "⏹️ Stopping batch processing..."
    
    def _batch_worker(self, input_dir, prompt_file, task, direction, expand_ratio):
        """실제 배치 처리 워커 - 다중 GPU 지원"""
        try:
            # GPU 개수 확인
            import torch
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
            nproc_per_node = min(gpu_count, 2)
            
            # ✅ 배치 모델 설정 가져오기 (UI에서 선택된 모델)
            batch_model = getattr(self, 'batch_model', None)
            if batch_model and hasattr(batch_model, 'value'):
                model_name = batch_model.value
            else:
                model_name = "vace-14B"  # 기본값
            
            # ✅ 모델별 체크포인트 디렉토리 설정
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
                
                print(f"🎬 Processing {filename} with {model_name} ({ckpt_dir})")
                print(f"📝 Prompt: {prompt}")
                
                # ✅ 다중 GPU 명령어 구성 - ckpt_dir 추가
                cmd = [
                    'torchrun', 
                    f'--nproc-per-node={nproc_per_node}',
                    'vace/vace_pipeline.py',
                    '--base', 'wan',
                    '--task', task,
                    '--video', video_file,
                    '--direction', ','.join(direction) if direction else 'left,right',
                    '--expand_ratio', str(expand_ratio),
                    '--prompt', prompt,
                    '--base_seed', str(2025 + completed),
                    '--model_name', model_name,  # ✅ 모델명 추가
                    '--ckpt_dir', ckpt_dir,      # ✅ 체크포인트 디렉토리 추가
                    '--sample_steps', '50',
                    '--sample_guide_scale', '5.0'
                ]
                
                # 다중 GPU 최적화 옵션
                if nproc_per_node > 1:
                    cmd.extend([
                        '--dit_fsdp',
                        '--t5_fsdp',
                        '--ulysses_size', str(nproc_per_node),
                        '--ring_size', '1'
                    ])
                
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        cwd="/data/VACE",
                        # timeout=1800
                    )
                    
                    if result.returncode == 0:
                        completed += 1
                        print(f"✅ Completed {filename} ({completed}/{total_files})")
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