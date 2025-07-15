# 水下目标检测系统 README

## 项目概述

本项目是一个基于YOLOv8框架的水下生物目标检测系统，专门针对水下环境的特殊挑战（如光线散射、低对比度、水体浑浊等）进行了优化。系统包含从数据采集、预处理、模型训练到实时检测的全流程解决方案，支持四种常见水下生物（海参、海胆、扇贝、海星）的检测。

## 项目文件结构

```
underwater_detection/
├── data_preprocessing/          # 数据预处理模块
│   ├── video_frame_extractor.py       # 视频帧提取工具
│   ├── xml_to_yolo.py        # 标注格式转换器
│   └── data_preparation.py      # 数据集准备工具
│
├── models/                      # 模型相关
│   ├── train_model.py         # 模型训练器
│   └── underwater.yaml         # 数据集配置文件
│
├── detection/                   # 检测模块
│   └── underwater_detection.py   # 实时检测工具
│
│
├── docs/                        # 文档
│   └── README.md                # 项目说明文档
```

## 核心代码文件说明

### 1. `video_frame_extractor.py` - 视频帧提取工具

#### 功能概述
- 从水下视频中按指定间隔提取帧图像
- 自动调整图像尺寸并保持长宽比
- 支持批量处理多个视频文件

#### 主要类和方法
- `VideoProcessor`类：视频处理基类
  - `_setup()`: 初始化视频处理环境
  - `release()`: 释放视频资源
- `smart_resize()`: 智能缩放和填充图像
- `extract_frames()`: 主处理函数

#### 使用示例
```python
extract_frames(
    input_path="input/video.mp4",
    output_dir="output/frames",
    frame_interval=10,
    target_size=(640, 640),
    skip_seconds=5,
    max_frames=1000
)
```

### 2. `xml_to_yolo.py` - 标注格式转换器

#### 功能概述
- 将PASCAL VOC格式(XML)转换为YOLO格式(TXT)
- 支持批量转换整个目录的标注文件
- 自动过滤无效标注

#### 主要类和方法
- `YOLOConverter`类：
  - `CLASS_MAP`: 类别映射字典
  - `convert_file()`: 单个文件转换
  - `batch_convert()`: 批量转换

#### 使用示例
```python
batch_convert(
    xml_dir="data/annotations/xml",
    output_dir="data/annotations/yolo"
)
```

### 3. `data_preparation.py` - 数据集准备工具

#### 功能概述
- 自动划分训练集/验证集/测试集
- 处理噪声数据(u前缀文件)
- 生成YOLO格式数据集结构
- 创建数据集配置文件

#### 主要功能函数
- `parse_xml_annotation()`: 解析XML标注
- `clean_noisy_data()`: 清洗噪声数据
- `save_yolo_annotation()`: 保存YOLO格式标注
- `prepare_dataset()`: 主处理函数
- `create_dataset_yaml()`: 创建配置文件

#### 使用说明
直接运行脚本即可自动处理数据集：
```bash
python dataset_preparer.py
```

### 4. `train_model.py` - 模型训练器

#### 功能概述
- 基于YOLOv8的自定义训练
- 实时监控训练指标
- 自动保存最佳模型
- 绘制损失曲线

#### 主要类和方法
- `UnderwaterDetectorTrainer`类：
  - `_check_environment()`: 检查训练环境
  - `_setup_model()`: 初始化模型
  - `train()`: 执行训练

#### 训练配置示例
```python
train_config = {
    'data': "config/underwater.yaml",
    'epochs': 100,
    'batch': 16,
    'imgsz': 640,
    'optimizer': 'AdamW',
    'lr0': 0.01,
    'cos_lr': True,
    'box': 7.5,  # 调整box损失权重
    'cls': 1.5   # 加强分类损失
}
```

### 5. `underwater_detection.py` - 实时检测工具

#### 功能概述
- 水下图像实时增强
- 目标检测与跟踪
- 性能统计与报告

#### 主要类和方法
- `UnderwaterEnhancer`类：
  - `remove_glare()`: 去除眩光
- `run_detection()`: 主检测函数

#### 使用示例
```python
run_detection(
    video_path="input/test_video.mp4",
    model_path="models/best.pt"
)
```

## 环境要求

- Python 3.7+
- OpenCV 4.5+
- PyTorch 1.10+
- Ultralytics YOLOv8
- CUDA 11.3+ (推荐)

安装依赖：
```bash
pip install -r requirements.txt
```

## 使用流程

1. 准备数据：
   ```bash
   python data_preprocessing/video_processor.py
   python data_preprocessing/yolo_converter.py
   python data_preprocessing/dataset_preparer.py
   ```

2. 训练模型：
   ```bash
   python models/model_trainer.py
   ```

3. 运行检测：
   ```bash
   python detection/underwater_detector.py
   ```

## 注意事项

1. 噪声数据(u前缀文件)需要特殊处理，建议先检查数据质量
2. 水下目标通常较小，建议使用高分辨率(640x640或更高)
3. 训练时建议使用GPU加速
4. 实际部署时可能需要根据具体水下环境调整增强参数

## 性能优化建议

1. 针对不同水域环境调整图像增强参数
2. 增加数据增强策略(如随机旋转、色彩抖动)
3. 使用更大的模型(yolov8m/yolov8l)提高检测精度
4. 使用TensorRT加速推理