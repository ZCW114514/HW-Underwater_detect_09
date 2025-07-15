import cv2
import numpy as np
import time
from ultralytics import YOLO

class UnderwaterEnhancer:
    """水下图像增强处理器"""
    @staticmethod
    def remove_glare(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
        frame[mask > 0] = 0
        
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)

def run_detection(video_path, model_path):
    """执行水下目标检测"""
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    fps_list = []
    conf_list = []
    start_time = time.time()
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        processed = UnderwaterEnhancer.remove_glare(frame)
        
        detect_start = time.time()
        results = model(processed, verbose=False)
        fps_list.append(1/(time.time() - detect_start))
        
        if results[0].boxes.conf.numel() > 0:
            conf_list.append(results[0].boxes.conf.mean().item())
        
        # 显示处理效果
        combined = np.hstack([frame, processed])
        cv2.imshow("Detection Results", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    duration = time.time() - start_time
    print("\n===== 检测报告 =====")
    print(f"总帧数: {frame_count}")
    print(f"平均帧率: {frame_count/duration:.2f} FPS")
    print(f"平均置信度: {np.mean(conf_list):.4f}" if conf_list else "未检测到目标")

if __name__ == "__main__":
    run_detection(
        video_path=r"F:\HW-Underwater_detect_09\output",
        model_path=r"F:\HW-Underwater_detect_09\models\yolov8s_underwater.pt"
    )