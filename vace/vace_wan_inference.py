# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import time
from datetime import datetime
import logging
import os
import sys
import warnings
import threading
import gc

warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
from PIL import Image

import wan
from wan.utils.utils import cache_video, cache_image, str2bool

from models.wan import WanVace
from models.wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from annotators.utils import get_annotator

# 전역 모델 캐시
_MODEL_CACHE = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TIMEOUT = int(os.getenv("MODEL_CACHE_TIMEOUT", "2000"))  # 1800=ㅏ30분

def get_cached_model(**model_kwargs):
    """캐시된 모델 반환 또는 새로 로드"""
    import hashlib
    
    # 캐시 키 생성 (device_id, rank 제외)
    key_parts = []
    for k, v in sorted(model_kwargs.items()):
        if k not in ['device_id', 'rank']:
            key_parts.append(f"{k}:{v}")
    cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()[:16]
    
    current_time = time.time()
    
    with _CACHE_LOCK:
        # 기존 캐시 확인
        if cache_key in _MODEL_CACHE:
            cache_data = _MODEL_CACHE[cache_key]
            if current_time - cache_data['last_used'] < _CACHE_TIMEOUT:
                cache_data['last_used'] = current_time
                logging.info("Using cached model")
                return cache_data['model']
            else:
                # 만료된 캐시 삭제
                del _MODEL_CACHE[cache_key]
                gc.collect()
                torch.cuda.empty_cache()
                logging.info("Expired model removed from cache")
    
    # 새 모델 로드
    logging.info("Loading new model...")
    model = WanVace(**model_kwargs)
    
    with _CACHE_LOCK:
        _MODEL_CACHE[cache_key] = {
            'model': model,
            'last_used': current_time
        }
    
    return model

EXAMPLE_PROMPT = {
    "vace-1.3B": {
        "src_ref_images": 'assets/images/girl.png,assets/images/snake.png',
        "prompt": "在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏。她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上洋溢着灿烂的笑容。蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽默。小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。周围五彩斑斓的灯笼和彩带装饰着环境，阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年氛围。"
    },
    "vace-14B": {
        "src_ref_images": 'assets/images/girl.png,assets/images/snake.png',
        "prompt": "在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏。她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上洋溢着灿烂的笑容。蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽默。小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。周围五彩斑斓的灯笼和彩带装饰着环境，阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年氛围。"
    }
}


def validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.model_name in WAN_CONFIGS, f"Unsupport model name: {args.model_name}"
    assert args.model_name in EXAMPLE_PROMPT, f"Unsupport model name: {args.model_name}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 50

    if args.sample_shift is None:
        args.sample_shift = 16

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 81

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.model_name], f"Unsupport size {args.size} for model name {args.model_name}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.model_name])}"
    return args


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vace-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The model name to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="720p",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default='models/Wan2.1-VACE-14B/',
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--src_video",
        type=str,
        default=None,
        help="The file of the source video. Default None.")
    parser.add_argument(
        "--src_mask",
        type=str,
        default=None,
        help="The file of the source mask. Default None.")
    parser.add_argument(
        "--src_ref_images",
        type=str,
        default=None,
        help="The file list of the source reference images. Separated by ','. Default None.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default=None,
        help="The negative prompt to avoid unwanted features. Will be combined with the model's default negative prompt.")
    parser.add_argument(
        "--use_prompt_extend",
        default='plain',
        choices=['plain', 'wan_zh', 'wan_en', 'wan_zh_ds', 'wan_en_ds'],
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=2025,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=6.0,
        help="Classifier free guidance scale.")
    return parser


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def main(args):
    args = argparse.Namespace(**args) if isinstance(args, dict) else args
    args = validate_args(args)

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (initialize_model_parallel,
                                             init_distributed_environment)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    if args.use_prompt_extend and args.use_prompt_extend != 'plain':
        prompt_expander = get_annotator(config_type='prompt', config_task=args.use_prompt_extend, return_dict=False)

    cfg = WAN_CONFIGS[args.model_name]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`num_heads` must be divisible by `ulysses_size`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.model_name]["prompt"]
        args.src_video = EXAMPLE_PROMPT[args.model_name].get("src_video", None)
        args.src_mask = EXAMPLE_PROMPT[args.model_name].get("src_mask", None)
        args.src_ref_images = EXAMPLE_PROMPT[args.model_name].get("src_ref_images", None)

    logging.info(f"Input prompt: {args.prompt}")
    if args.use_prompt_extend and args.use_prompt_extend != 'plain':
        logging.info("Extending prompt ...")
        if rank == 0:
            prompt = prompt_expander.forward(args.prompt)
            logging.info(f"Prompt extended from '{args.prompt}' to '{prompt}'")
            input_prompt = [prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")

    original_neg_prompt = cfg.sample_neg_prompt  # Store original
    if args.neg_prompt:
        # Temporarily modify the config's negative prompt
        cfg.sample_neg_prompt = f"{args.neg_prompt}, {original_neg_prompt}"
        logging.info(f"User negative prompt: {args.neg_prompt}")
    logging.info(f"Final negative prompt: {cfg.sample_neg_prompt}")

    logging.info("Getting model from cache or loading new model...")
    wan_vace = get_cached_model(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )
    ### Cha's edit
    # wan_vace = torch.compile(wan_vace, mode='default', fullgraph=True)
    ###

    actual_fps = cfg.sample_fps  # Default FPS
    if args.src_video and os.path.exists(args.src_video):
        import cv2
        cap = cv2.VideoCapture(args.src_video)
        if cap.isOpened():
            input_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if input_fps > 0:
                actual_fps = input_fps
                logging.info(f"Using input video FPS: {actual_fps}")
            else:
                logging.warning(f"Could not get FPS from input video, using default: {actual_fps}")
            cap.release()
        else:
            logging.warning(f"Could not open input video, using default FPS: {actual_fps}")
    
    # Update the video processor with actual FPS
    wan_vace.vid_proc.min_fps = actual_fps
    wan_vace.vid_proc.max_fps = actual_fps
    logging.info(f"Video processor FPS set to: {actual_fps}")

    src_video, src_mask, src_ref_images = wan_vace.prepare_source([args.src_video],
                                                                  [args.src_mask],
                                                                  [None if args.src_ref_images is None else args.src_ref_images.split(',')],
                                                                  args.frame_num, SIZE_CONFIGS[args.size], device)
    # TODO: 한번에 처리가 되지않는 긴 영상의 자연스러운 확장에 쓰임
    # import cv2

    # prev_video_path = "/data/VACE/results/vace_wan_1.3b/2025-05-08-07-26-08/out_video.mp4"
    # cap = cv2.VideoCapture(prev_video_path)

    # if not cap.isOpened():
    #     logging.error(f"영상 열기에 실패했습니다: {prev_video_path}")
    # else:
    #     logging.info(f"영상 열기 성공: {prev_video_path}")

    #     # 총 프레임 수 확인
    #     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     logging.info(f"총 프레임 수: {total_frames}")

    #     if total_frames > 0:
    #         # 1. 원본 영상 프레임 수 확인 및 저장
    #         src_frames = src_video[0].shape[1]  # 원본 프레임 수
    #         front_frames = 41  # 앞으로 밀 프레임 수
    #         insert_frames = 40  # 삽입할 이전 영상의 프레임 수
            
    #         logging.info(f"원본 영상 프레임 수: {src_frames}, 앞으로 밀 프레임: {front_frames}, 삽입할 프레임: {insert_frames}")
            
    #         # 2. 원본 영상의 앞부분 저장
    #         original_front = src_video[0][:, :front_frames].clone()
    #         logging.info(f"저장된 원본 앞부분 크기: {original_front.shape}")
            
    #         # 3. 이전 영상의 뒷부분 프레임 추출
    #         frames_to_extract = min(insert_frames, total_frames)  # 추출할 프레임 수
    #         extracted_frames = []
            
    #         # 마지막 N개 프레임 추출
    #         for i in range(frames_to_extract):
    #             frame_idx = total_frames - frames_to_extract + i
    #             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    #             ret, frame = cap.read()
                
    #             if ret:
    #                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #                 frame_tensor = torch.from_numpy(frame).float() / 255.0
    #                 frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC -> CHW
    #                 frame_tensor = frame_tensor.mul_(2).sub_(1)  # [0,1] -> [-1,1]
    #                 extracted_frames.append(frame_tensor)
    #             else:
    #                 logging.error(f"프레임 {frame_idx} 읽기 실패")
            
    #         if extracted_frames:
    #             # 추출한 프레임들을 하나의 텐서로 결합
    #             extracted_tensor = torch.stack(extracted_frames, dim=1).to(device)
    #             logging.info(f"이전 영상 추출 프레임 텐서 크기: {extracted_tensor.shape}")
                
    #             # 4. 새 영상 조합: 이전 영상의 뒷부분 + 원본 영상의 앞부분
    #             new_video = torch.cat([
    #                 extracted_tensor,         # 이전 영상의 뒷부분 (40프레임)
    #                 original_front            # 원본 영상의 앞부분 (41프레임)
    #             ], dim=1)
                
    #             # 필요시 길이 조정 (프레임 수 맞추기)
    #             if new_video.shape[1] != src_frames:
    #                 if new_video.shape[1] > src_frames:
    #                     new_video = new_video[:, :src_frames]  # 길이 줄이기
    #                 else:
    #                     # 길이 늘리기 (필요한 경우)
    #                     padding = src_frames - new_video.shape[1]
    #                     padding_frames = src_video[0][:, -padding:].clone()
    #                     new_video = torch.cat([new_video, padding_frames], dim=1)
                
    #             # 5. 최종 적용
    #             src_video[0] = new_video
    #             logging.info(f"영상 재구성 완료. 최종 영상 크기: {src_video[0].shape}")
    #         else:
    #             logging.error("이전 영상에서 추출된 프레임이 없습니다.")
    #     else:
    #         logging.error("영상의 프레임 수가 0입니다.")

    # # 자원 해제
    # cap.release()
    # 기존 주석처리된 부분을 다음과 같이 수정

    # # TODO: 영상을 확장했는데 초반은 마음에 드는데 후반부가 마음에 안들 때 해당 영상의 앞부분을 가이드로 줘서 재생성할 때 쓰임
    # import cv2

    # # 1. 이전 결과 영상 경로 업데이트
    # prev_video_path = "/data/VACE/results/61_s.mp4" #"/data/VACE/results/vace-14B/2025-05-26-19-21-56/out_video.mp4"
    # cap = cv2.VideoCapture(prev_video_path)

    # if not cap.isOpened():
    #     logging.error(f"영상 열기에 실패했습니다: {prev_video_path}")
    # else:
    #     logging.info(f"영상 열기 성공: {prev_video_path}")

    #     # 총 프레임 수 확인
    #     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     fps = cap.get(cv2.CAP_PROP_FPS)
    #     logging.info(f"총 프레임 수: {total_frames}, FPS: {fps}")

    #     if total_frames > 0:
    #         # 2. 프레임 수 계산 (0.82초까지 사용)
    #         src_frames = src_video[0].shape[1]  # 원본 프레임 수 (81프레임)
    #         guide_duration_seconds = 2.6  # 가이드로 사용할 시간 (초)
    #         target_fps = 24  # 목표 FPS (VACE 기본값)
            
    #         # 가이드로 사용할 프레임 수 계산
    #         guide_frames = int(guide_duration_seconds * target_fps)  # 0.82초* 24프레임/초 ≈ 20프레임
    #         remaining_frames = src_frames - guide_frames  # 나머지 61프레임
            
    #         logging.info(f"원본 영상 프레임 수: {src_frames}")
    #         logging.info(f"가이드 프레임 수: {guide_frames}")
    #         logging.info(f"새로 생성할 프레임 수: {remaining_frames}")
            
    #         # 3. 이전 영상에서 가이드 프레임 추출
    #         frames_to_extract = min(guide_frames, total_frames)
    #         extracted_frames = []
            
    #         # 앞부분 프레임 추출 (0.82초까지)
    #         for i in range(frames_to_extract):
    #             frame_idx = i
    #             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    #             ret, frame = cap.read()
    #             if ret:
    #                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #                 frame_tensor = torch.from_numpy(frame).float() / 255.0
    #                 frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC -> CHW
    #                 frame_tensor = frame_tensor.mul_(2).sub_(1)  # [0,1] -> [-1,1]
    #                 extracted_frames.append(frame_tensor)
    #             else:
    #                 logging.error(f"프레임 {frame_idx} 읽기 실패")
            
    #         if extracted_frames:
    #             # 4. 추출한 프레임들을 텐서로 변환
    #             extracted_tensor = torch.stack(extracted_frames, dim=1).to(device)
    #             logging.info(f"이전 영상 추출 프레임 텐서 크기: {extracted_tensor.shape}")
                
    #             # 5. 원본 영상의 뒷부분 (새로 생성될 부분의 초기값)
    #             original_back = src_video[0][:, guide_frames:].clone()

    #             # 6. 새 영상 조합: 이전 영상의 좋은 부분(가이드) + 원본 영상의 뒷부분(생성 대상)
    #             new_video = torch.cat([
    #                 extracted_tensor,     # 이전 영상의 좋은 부분 (40프레임) - 가이드 역할
    #                 original_back         # 원본 영상의 뒷부분 (41프레임) - 새로 생성될 부분
    #             ], dim=1)
                
    #             # 7. CRITICAL: Update the mask to match the guidance
    #             # The mask should NOT generate (mask=1) the guided portion in expanded areas
    #             original_mask_back = src_mask[0][:, guide_frames:].clone()
                
    #             # DEBUG: 원본 마스크 상태 확인
    #             logging.info(f"=== 마스크 디버깅 시작 ===")
    #             logging.info(f"원본 전체 마스크 크기: {src_mask[0].shape}")
    #             logging.info(f"원본 마스크 뒷부분 크기: {original_mask_back.shape}")
                
    #             # 원본 마스크의 생성 영역 확인 (mask=1인 영역)
    #             original_mask_sum = (src_mask[0] > 0.5).sum(dim=(0, 2, 3))  # 각 프레임별 생성 픽셀 수
    #             original_back_mask_sum = (original_mask_back > 0.5).sum(dim=(0, 2, 3))
                
    #             logging.info(f"원본 마스크 각 프레임별 생성 픽셀 수 (처음 10개): {original_mask_sum[:10].tolist()}")
    #             logging.info(f"원본 마스크 뒷부분 각 프레임별 생성 픽셀 수 (처음 10개): {original_back_mask_sum[:10].tolist()}")

    #             # FIXED: Check and match channel dimensions
    #             mask_channels = original_mask_back.shape[0]  # Get actual mask channel count
    #             logging.info(f"마스크 채널 수: {mask_channels}, 비디오 채널 수: {extracted_tensor.shape[0]}")

    #             # CORRECTED: Create guided mask with ALL ZEROS (no generation for guided frames)
    #             # 가이드 프레임에서는 아무것도 생성하지 않음 (모두 0으로 설정)
    #             guided_mask = torch.zeros((mask_channels, extracted_tensor.shape[1], 
    #                                     extracted_tensor.shape[2], extracted_tensor.shape[3]), 
    #                                     device=extracted_tensor.device, dtype=original_mask_back.dtype)
                
    #             logging.info(f"가이드 마스크 생성 - 전체가 0 (생성하지 않음): {(guided_mask == 0).all()}")
                
    #             # Combine guided mask with original mask for remaining frames
    #             new_mask = torch.cat([
    #                 guided_mask,          # Guided portion: NO generation (all zeros)
    #                 original_mask_back    # Remaining portion: use original mask
    #             ], dim=1)
                
    #             # DEBUG: 결합된 마스크 확인
    #             new_mask_sum = (new_mask > 0.5).sum(dim=(0, 2, 3))
    #             logging.info(f"결합된 마스크 각 프레임별 생성 픽셀 수:")
    #             logging.info(f"  - 가이드 부분 (0~{guide_frames-1}): {new_mask_sum[:guide_frames].tolist()}")
    #             logging.info(f"  - 생성 부분 ({guide_frames}~{guide_frames+10}): {new_mask_sum[guide_frames:guide_frames+10].tolist()}")
                
    #             # 전체 마스크 통계
    #             total_guided_pixels = (guided_mask > 0.5).sum().item()
    #             total_original_back_pixels = (original_mask_back > 0.5).sum().item()
    #             total_new_mask_pixels = (new_mask > 0.5).sum().item()
                
    #             logging.info(f"전체 생성 픽셀 수:")
    #             logging.info(f"  - 가이드 마스크: {total_guided_pixels}")
    #             logging.info(f"  - 원본 뒷부분 마스크: {total_original_back_pixels}")
    #             logging.info(f"  - 결합된 마스크: {total_new_mask_pixels}")
    #             logging.info(f"=== 마스크 디버깅 종료 ===")

    #             # 8. 길이 검증 및 조정
    #             if new_video.shape[1] != src_frames:
    #                 if new_video.shape[1] > src_frames:
    #                     new_video = new_video[:, :src_frames]
    #                     new_mask = new_mask[:, :src_frames]
    #                 else:
    #                     # 부족한 경우 패딩 추가
    #                     padding = src_frames - new_video.shape[1]
    #                     padding_frames = src_video[0][:, -padding:].clone()
    #                     padding_masks = src_mask[0][:, -padding:].clone()
    #                     new_video = torch.cat([new_video, padding_frames], dim=1)
    #                     new_mask = torch.cat([new_mask, padding_masks], dim=1)
                
    #             # 9. 최종 적용
    #             src_video[0] = new_video
    #             src_mask[0] = new_mask  # IMPORTANT: Update the mask too
                
    #             logging.info(f"영상 재구성 완료. 최종 영상 크기: {src_video[0].shape}")
    #             logging.info(f"마스크 재구성 완료. 최종 마스크 크기: {src_mask[0].shape}")
    #             logging.info(f"가이드 프레임: {guide_frames}, 새로 생성될 프레임: {remaining_frames}")
    #         else:
    #             logging.error("이전 영상에서 추출된 프레임이 없습니다.")
    #     else:
    #         logging.error("영상의 프레임 수가 0입니다.")

    # # 자원 해제
    # cap.release()
    logging.info(f"Generating video...")
    video = wan_vace.generate(
        args.prompt,
        src_video,
        src_mask,
        src_ref_images,
        size=SIZE_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model)

    # CLEANUP: Restore original negative prompt
    cfg.sample_neg_prompt = original_neg_prompt

    try:
        ret_data = {}
        if rank == 0:
            if args.save_dir is None:
                # 입력 파일명에서 디렉토리명 생성
                input_name = ""
                if args.src_video:
                    # 1. 폴더명에서 원본 파일명 추출
                    dir_path = os.path.dirname(args.src_video)
                    folder_name = os.path.basename(dir_path)  # "272_2025-06-16-09-45-06"
                    
                    # 타임스탬프 제거하여 원본 파일명 추출
                    import re
                    pattern = r'_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$'
                    original_name = re.sub(pattern, '', folder_name)  # "272"
                    
                    # 2. 파일명에서 작업명 추출
                    filename = os.path.basename(args.src_video)  # "src_video-outpainting.mp4"
                    filename_without_ext = os.path.splitext(filename)[0]  # "src_video-outpainting"
                    
                    # "src_video-" 제거하여 작업명만 추출
                    if filename_without_ext.startswith('src_video-'):
                        task_name = filename_without_ext[len('src_video-'):]  # "outpainting"
                    else:
                        task_name = filename_without_ext
                    
                    # 3. 현재 시간 생성
                    import time
                    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
                    
                    # 4. 최종 조합
                    input_name = f"{original_name}-{task_name}-{current_time}"  # "272-outpainting-2025-06-16-15-30-45"
                    
                    print(f"Input name: {input_name}")
                    save_dir = os.path.join('results', args.model_name, input_name)
            else:
                save_dir = args.save_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            if args.save_file is not None:
                save_file = args.save_file
            else:
                save_file = os.path.join(save_dir, 'out_video.mp4')

            cache_video(
                tensor=video[None],
                save_file=save_file,
                fps=actual_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
            logging.info(f"Saving generated video to {save_file}")
            ret_data['out_video'] = save_file

            save_file = os.path.join(save_dir, 'src_video.mp4')
            cache_video(
                tensor=src_video[0][None],
                save_file=save_file,
                fps=actual_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
            logging.info(f"Saving src_video to {save_file}")
            ret_data['src_video'] = save_file

            save_file = os.path.join(save_dir, 'src_mask.mp4')
            cache_video(
                tensor=src_mask[0][None],
                save_file=save_file,
                fps=actual_fps,
                nrow=1,
                normalize=True,
                value_range=(0, 1))
            logging.info(f"Saving src_mask to {save_file}")
            ret_data['src_mask'] = save_file

            if src_ref_images[0] is not None:
                for i, ref_img in enumerate(src_ref_images[0]):
                    save_file = os.path.join(save_dir, f'src_ref_image_{i}.png')
                    cache_image(
                        tensor=ref_img[:, 0, ...],
                        save_file=save_file,
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1))
                    logging.info(f"Saving src_ref_image_{i} to {save_file}")
                    ret_data[f'src_ref_image_{i}'] = save_file
        logging.info("Finished.")
        return ret_data
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        raise
    finally:
        # 분산 처리 정리
        if dist.is_initialized():
            try:
                logging.info("Destroying process group...")
                dist.destroy_process_group()
            except Exception as e:
                logging.warning(f"Error during cleanup: {e}")

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)