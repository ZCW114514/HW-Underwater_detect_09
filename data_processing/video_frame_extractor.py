import os
import cv2
import numpy as np
from tqdm import tqdm

class VideoProcessor:
    """视频处理基础类"""
    def __init__(self, input_path, output_dir):
        self.input_path = input_path
        self.output_dir = output_dir
        self.cap = None
        
    def _setup(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.input_path}")
        return self.cap.get(cv2.CAP_PROP_FPS), int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def release(self):
        if self.cap:
            self.cap.release()

def smart_resize(frame, target_size=(640, 640)):
    """智能缩放和填充图像"""
    h, w = frame.shape[:2]
    scale = min(target_size[0]/w, target_size[1]/h)
    resized = cv2.resize(frame, None, fx=scale, fy=scale)
    
    padded = np.full((*target_size[::-1], 3), 114, dtype=np.uint8)
    pad_h, pad_w = [(target_size[i] - resized.shape[i])//2 for i in (1, 0)]
    padded[pad_h:pad_h+resized.shape[0], pad_w:pad_w+resized.shape[1]] = resized
    return padded

def extract_frames(input_path, output_dir, frame_interval=10, target_size=(640, 640), skip_seconds=0, max_frames=None):
    """主处理函数"""
    processor = VideoProcessor(input_path, output_dir)
    fps, total_frames = processor._setup()
    
    skip_frames = int(skip_seconds * fps)
    processor.cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    pbar = tqdm(total=min(max_frames or total_frames, total_frames - skip_frames), 
               desc="提取进度", unit="帧")
    
    saved_count = 0
    for frame_count in range(skip_frames, total_frames):
        ret, frame = processor.cap.read()
        if not ret or (max_frames and saved_count >= max_frames):
            break
            
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_dir, f"{base_name}_frame{frame_count:06d}.jpg")
            cv2.imwrite(output_path, smart_resize(frame, target_size), 
                       [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1
            
        pbar.update(1)
    
    pbar.close()
    processor.release()
    print(f"\n处理完成！共保存 {saved_count} 帧图像到: {output_dir}")

if __name__ == "__main__":
    extract_frames(
        input_path=r"E:\BaiduNetdiskDownload\YN020013.MP4",
        output_dir=r"E:\BaiduNetdiskDownload\train1\deal",
        frame_interval=10,
        target_size=(640, 640)
    )