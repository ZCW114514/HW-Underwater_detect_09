import os
import xml.etree.ElementTree as ET
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 数据集路径配置
DATA_ROOT = '/root/Downloads/HW-Underwater_detect_09/'
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
TEST_A_IMAGE = os.path.join(DATA_ROOT, 'test-A-image')
TEST_A_BOX = os.path.join(DATA_ROOT, 'test-A-box')
TEST_B_IMAGE = os.path.join(DATA_ROOT, 'test-B-image')
TEST_B_BOX = os.path.join(DATA_ROOT, 'test-B-box')
OUTPUT_DIR = os.path.join(DATA_ROOT, 'yolo_dataset')

# 目标类别映射
CLASS_MAP = {
    'holothurian': 0,  # 海参
    'echinus': 1,      # 海胆
    'scallop': 2,      # 扇贝
    'starfish': 3      # 海星
}

def parse_xml_annotation(xml_path):
    """解析XML标注文件"""
    if not os.path.exists(xml_path):
        print(f"警告：标注文件 {xml_path} 不存在")
        return []
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        annotations = []
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            
            # 忽略水草类别
            if cls_name == 'waterweeds':
                continue
                
            # 只处理四类目标生物
            if cls_name not in CLASS_MAP:
                print(f"警告：发现未知类别 '{cls_name}'，跳过")
                continue
                
            bbox = obj.find('bndbox')
            try:
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
            except (AttributeError, ValueError):
                print(f"警告：标注文件 {xml_path} 中的边界框格式错误，跳过")
                continue
            
            # 确保边界框有效
            if xmin >= xmax or ymin >= ymax:
                print(f"警告：无效边界框 ({xmin},{ymin},{xmax},{ymax})，跳过")
                continue
                
            annotations.append({
                'class': cls_name,
                'class_id': CLASS_MAP[cls_name],
                'bbox': [xmin, ymin, xmax, ymax]
            })
        
        return annotations
    except ET.ParseError:
        print(f"错误：无法解析XML文件 {xml_path}")
        return []

def process_image_and_annotations(image_path, box_dir, output_image_dir, output_label_dir):
    """处理单个图像及其标注"""
    img_name = os.path.basename(image_path)
    base_name = os.path.splitext(img_name)[0]
    xml_path = os.path.join(box_dir, f"{base_name}.xml")
    
    # 区分干净数据和噪声数据
    if img_name.startswith('u'):
        # 噪声数据处理
        img, annotations = clean_noisy_data(image_path, xml_path)
        if img is None or not annotations:
            return False
        # 保存处理后的图像
        cv2.imwrite(os.path.join(output_image_dir, img_name), img)
    else:
        # 干净数据处理
        if not os.path.exists(xml_path):
            return False
        annotations = parse_xml_annotation(xml_path)
        if not annotations:
            return False
        # 复制图像
        shutil.copy(image_path, os.path.join(output_image_dir, img_name))
    
    # 保存YOLO格式标注
    label_path = os.path.join(output_label_dir, f"{base_name}.txt")
    save_yolo_annotation(image_path, label_path, annotations)
    return True

def clean_noisy_data(img_path, xml_path):
    """清洗噪声标注数据（u前缀文件）"""
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"警告：无法读取图像 {img_path}，跳过")
        return None, []
    
    height, width = img.shape[:2]
    
    # 解析原始标注
    orig_annotations = parse_xml_annotation(xml_path)
    
    # 噪声数据处理策略
    cleaned_annotations = []
    for ann in orig_annotations:
        xmin, ymin, xmax, ymax = ann['bbox']
        
        # 确保坐标在图像范围内
        xmin = max(0, min(xmin, width - 1))
        ymin = max(0, min(ymin, height - 1))
        xmax = max(1, min(xmax, width))
        ymax = max(1, min(ymax, height))
        
        # 计算边界框面积
        bbox_area = (xmax - xmin) * (ymax - ymin)
        img_area = width * height
        
        # 过滤过小或过大的边界框
        if bbox_area < 100:  # 小于100像素
            print(f"警告：跳过过小目标 ({bbox_area}像素)")
            continue
        if bbox_area > img_area * 0.5:  # 大于图像面积的50%
            print(f"警告：跳过过大目标 ({bbox_area}像素)")
            continue
            
        # 更新边界框
        ann['bbox'] = [xmin, ymin, xmax, ymax]
        cleaned_annotations.append(ann)
    
    return img, cleaned_annotations

def save_yolo_annotation(img_path, output_path, annotations):
    """保存YOLO格式的标注文件"""
    # 获取图像尺寸
    img = cv2.imread(img_path)
    if img is None:
        print(f"错误：无法读取图像 {img_path} 以获取尺寸")
        return
    
    height, width = img.shape[:2]
    
    with open(output_path, 'w') as f:
        for ann in annotations:
            class_id = ann['class_id']
            xmin, ymin, xmax, ymax = ann['bbox']
            
            # 转换为YOLO格式 (归一化中心坐标和宽高)
            cx = (xmin + xmax) / 2 / width
            cy = (ymin + ymax) / 2 / height
            nw = (xmax - xmin) / width
            nh = (ymax - ymin) / height
            
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

def prepare_dataset():
    """准备YOLO格式数据集"""
    # 创建输出目录结构
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', 'testA'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', 'testA'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', 'testB'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', 'testB'), exist_ok=True)
    
    # 处理训练集数据
    train_images = [f for f in os.listdir(os.path.join(TRAIN_DIR, 'image')) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not train_images:
        print(f"错误：在 {TRAIN_DIR} 中没有找到任何图片文件！")
        return
    
    print(f"找到 {len(train_images)} 张训练图片")
    
    # 分类干净数据和噪声数据
    clean_data = []
    noisy_data = []
    for img_name in train_images:
        if img_name.startswith('c'):
            clean_data.append(img_name)
        elif img_name.startswith('u'):
            noisy_data.append(img_name)
        else:
            print(f"警告：图片 {img_name} 没有c或u前缀，将作为干净数据处理")
            clean_data.append(img_name)
    
    print(f"干净数据: {len(clean_data)} 张, 噪声数据: {len(noisy_data)} 张")
    
    # 划分训练集和验证集 (8:2)
    train_files, val_files = train_test_split(clean_data, test_size=0.2, random_state=42)
    
    # 处理训练集干净数据
    for img_name in tqdm(train_files, desc="处理训练集干净数据"):
        img_path = os.path.join(TRAIN_DIR, 'image', img_name)
        box_dir = os.path.join(TRAIN_DIR, 'box')
        process_image_and_annotations(
            img_path, box_dir, 
            os.path.join(OUTPUT_DIR, 'images', 'train'),
            os.path.join(OUTPUT_DIR, 'labels', 'train')
        )
    
    # 处理验证集干净数据
    for img_name in tqdm(val_files, desc="处理验证集干净数据"):
        img_path = os.path.join(TRAIN_DIR, 'image', img_name)
        box_dir = os.path.join(TRAIN_DIR, 'box')
        process_image_and_annotations(
            img_path, box_dir, 
            os.path.join(OUTPUT_DIR, 'images', 'val'),
            os.path.join(OUTPUT_DIR, 'labels', 'val')
        )
    
    # 处理噪声数据（全部加入训练集）
    for img_name in tqdm(noisy_data, desc="处理噪声数据"):
        img_path = os.path.join(TRAIN_DIR, 'image', img_name)
        box_dir = os.path.join(TRAIN_DIR, 'box')
        process_image_and_annotations(
            img_path, box_dir, 
            os.path.join(OUTPUT_DIR, 'images', 'train'),
            os.path.join(OUTPUT_DIR, 'labels', 'train')
        )
    
    # 处理测试集A
    test_a_images = [f for f in os.listdir(TEST_A_IMAGE) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img_name in tqdm(test_a_images, desc="处理测试集A"):
        img_path = os.path.join(TEST_A_IMAGE, img_name)
        process_image_and_annotations(
            img_path, TEST_A_BOX,
            os.path.join(OUTPUT_DIR, 'images', 'testA'),
            os.path.join(OUTPUT_DIR, 'labels', 'testA')
        )
    
    # 处理测试集B
    test_b_images = [f for f in os.listdir(TEST_B_IMAGE) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img_name in tqdm(test_b_images, desc="处理测试集B"):
        img_path = os.path.join(TEST_B_IMAGE, img_name)
        process_image_and_annotations(
            img_path, TEST_B_BOX,
            os.path.join(OUTPUT_DIR, 'images', 'testB'),
            os.path.join(OUTPUT_DIR, 'labels', 'testB')
        )
    
    # 创建数据集配置文件
    create_dataset_yaml()

def create_dataset_yaml():
    """创建YOLOv5数据集配置文件"""
    yaml_content = f"""# YOLOv5 underwater dataset config
path: {OUTPUT_DIR}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
testA: images/testA  # testA images (relative to 'path')
testB: images/testB  # testB images (relative to 'path')

# Classes
names:
  0: holothurian
  1: echinus
  2: scallop
  3: starfish
"""
    with open(os.path.join(DATA_ROOT, 'underwater.yaml'), 'w') as f:
        f.write(yaml_content)
    
    print(f"数据集配置文件已保存到 {os.path.join(DATA_ROOT, 'underwater.yaml')}")

if __name__ == "__main__":
    prepare_dataset()