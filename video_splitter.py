import os
import math
import argparse
import subprocess
import json

def get_video_info(video_path):
    """비디오 파일의 정보를 가져옵니다 (길이, 해상도, 프레임 레이트 등)"""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    data = json.loads(result.stdout)
    
    # 비디오 정보 찾기
    video_stream = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    
    return {
        "duration": float(data["format"]["duration"]),
        "width": int(video_stream["width"]) if video_stream else 1920,
        "height": int(video_stream["height"]) if video_stream else 1080,
        "fps": eval(video_stream.get("r_frame_rate", "25/1")) if video_stream else 25
    }

def split_video_into_81frame_chunks(input_video_path, output_directory=None):
    """
    영상을 81프레임 단위 청크로 분할합니다. 마지막 청크가 81프레임보다 짧을 경우
    검은색 프레임으로 패딩하여 정확히 81프레임으로 만듭니다.
    """
    # 디렉토리와 파일명 가져오기
    input_dir, input_filename = os.path.split(input_video_path)
    filename_without_ext, ext = os.path.splitext(input_filename)
    
    # 출력 디렉토리 설정
    if output_directory is None:
        output_directory = input_dir
    
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_directory, exist_ok=True)
    
    # 비디오 정보 가져오기
    video_info = get_video_info(input_video_path)
    duration = video_info["duration"]
    width = video_info["width"]
    height = video_info["height"]
    fps = video_info["fps"]
    
    # 81프레임에 해당하는 시간 계산
    chunk_duration_seconds = 81 / fps  # 24fps 기준 3.375초
    
    # 필요한 청크 수 계산
    num_chunks = math.ceil(duration / chunk_duration_seconds)
    
    output_paths = []
    
    for i in range(num_chunks):
        start_time = i * chunk_duration_seconds
        actual_chunk_duration = min(chunk_duration_seconds, duration - start_time)
        
        # 출력 파일명 생성
        output_filename = f"{filename_without_ext}_chunk{i+1}{ext}"
        output_path = os.path.join(output_directory, output_filename)
        output_paths.append(output_path)
        
        # 청크가 81프레임보다 짧으면 패딩 추가
        if actual_chunk_duration < chunk_duration_seconds:
            temp_chunk = "temp_chunk.mp4"
            
            # 원본 청크 추출
            subprocess.run([
                "ffmpeg", "-y", "-ss", str(start_time), 
                "-i", input_video_path, 
                "-t", str(actual_chunk_duration),
                "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental",
                temp_chunk
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # 블랙 프레임 생성 (정확히 원본 비디오와 동일한 설정)
            black_duration = chunk_duration_seconds - actual_chunk_duration
            temp_black = "temp_black.mp4"
            subprocess.run([
                "ffmpeg", "-y", 
                "-f", "lavfi", 
                "-i", f"color=c=black:s={width}x{height}:r={fps}", 
                "-t", str(black_duration),
                "-c:v", "libx264", 
                temp_black
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # 두 파일 합치기
            file_list = "temp_file_list.txt"
            with open(file_list, "w") as f:
                f.write(f"file '{temp_chunk}'\nfile '{temp_black}'")
            
            subprocess.run([
                "ffmpeg", "-y", 
                "-f", "concat", 
                "-safe", "0", 
                "-i", file_list, 
                "-c", "copy", 
                output_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # 임시 파일 삭제
            os.remove(temp_chunk)
            os.remove(temp_black)
            os.remove(file_list)
        else:
            # 일반 청크 추출 (전체 81프레임)
            subprocess.run([
                "ffmpeg", "-y", "-ss", str(start_time), 
                "-i", input_video_path, 
                "-t", str(chunk_duration_seconds),
                "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental",
                output_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    return output_paths

def main():
    # CLI 인자 파서 설정
    parser = argparse.ArgumentParser(description="영상을 81프레임 단위 청크로 분할합니다")
    parser.add_argument("input_video", help="입력 영상 파일 경로")
    parser.add_argument("-o", "--output-dir", help="출력 청크를 저장할 디렉토리 (기본값: 입력 영상과 동일 디렉토리)")
    parser.add_argument("-v", "--verbose", action="store_true", help="상세 출력 활성화")
    
    # 인자 파싱
    args = parser.parse_args()
    
    # FFmpeg 유무 확인
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("오류: ffmpeg가 설치되어 있지 않습니다.")
        print("Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("CentOS/RHEL: sudo yum install ffmpeg")
        print("macOS (Homebrew): brew install ffmpeg")
        return
    
    # 영상 분할 함수 호출
    try:
        print(f"영상 분할 중 (81프레임 단위): {args.input_video}")
        output_videos = split_video_into_81frame_chunks(args.input_video, args.output_dir)
        
        # 결과 출력
        print(f"\n성공! {len(output_videos)}개의 영상 청크를 생성했습니다:")
        for path in output_videos:
            print(f"  - {path}")
        print(f"각 청크: 81프레임 (약 {81/24:.3f}초)")
    except Exception as e:
        print(f"오류 발생: {e}")

# 스크립트가 직접 실행될 때만 main() 함수 호출
if __name__ == "__main__":
    main()
