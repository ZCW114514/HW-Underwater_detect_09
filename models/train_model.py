from ultralytics import YOLO
import torch
import os
import matplotlib.pyplot as plt
from typing import Dict, Tuple

class UnderwaterDetectorTrainer:
    """水下目标检测模型训练器"""
    
    def __init__(self, config: Dict):
        """
        初始化训练器
        :param config: 训练配置字典
        """
        self.config = config
        self.loss_tracker = {
            'train': [],
            'val': [],
            'box': [],
            'cls': [],
            'dfl': []
        }
        
    def _check_environment(self):
        """检查训练环境"""
        print("="*50)
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"当前GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("警告: 将使用CPU训练，速度会显著降低")
        
    def _validate_dataset(self):
        """验证数据集配置"""
        assert os.path.exists(self.config['data']), f"dataset.yaml不存在: {self.config['data']}"
        
        # 从yaml中提取路径进行验证
        with open(self.config['data']) as f:
            data = f.read()
            assert "train:" in data and "val:" in data, "yaml文件缺少训练/验证路径"
            assert "names: ['holothurian', 'echinus', 'scallop', 'starfish']" in data, "类别名称不匹配"
        
        print("数据集验证通过！")
    
    def _setup_model(self) -> YOLO:
        """初始化YOLO模型"""
        model = YOLO("yolov8n.yaml")  # 从零开始构建
        model.model.nc = 4  # 强制设置为4类水下生物
        return model
    
    def _log_metrics(self, trainer):
        """记录训练指标"""
        metrics = trainer.validator.metrics
        epoch = trainer.epoch
        
        # 记录各项损失
        self.loss_tracker['train'].append(trainer.tloss)
        self.loss_tracker['val'].append(metrics.box_loss + metrics.cls_loss + metrics.dfl_loss)
        self.loss_tracker['box'].append(metrics.box_loss)
        self.loss_tracker['cls'].append(metrics.cls_loss)
        self.loss_tracker['dfl'].append(metrics.dfl_loss)
        
        # 打印当前epoch信息
        print(f"\nEpoch {epoch}/{self.config['epochs']}")
        print(f"Train Loss: {trainer.tloss:.4f} | Val Loss: {self.loss_tracker['val'][-1]:.4f}")
        print(f"Box Loss: {metrics.box_loss:.4f} | Cls Loss: {metrics.cls_loss:.4f} | DFL Loss: {metrics.dfl_loss:.4f}")
    
    def _plot_loss_curves(self, save_path: str):
        """绘制损失曲线"""
        plt.figure(figsize=(12, 8))
        
        # 主损失曲线
        plt.plot(self.loss_tracker['train'], label='Train Loss', color='blue')
        plt.plot(self.loss_tracker['val'], label='Validation Loss', color='orange')
        
        # 辅助损失曲线
        plt.plot(self.loss_tracker['box'], '--', label='Box Loss', alpha=0.5)
        plt.plot(self.loss_tracker['cls'], '--', label='Cls Loss', alpha=0.5)
        plt.plot(self.loss_tracker['dfl'], '--', label='DFL Loss', alpha=0.5)
        
        plt.title('Training Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        print(f"Loss曲线已保存到: {save_path}")
    
    def train(self) -> Tuple[YOLO, Dict]:
        """执行模型训练"""
        self._check_environment()
        self._validate_dataset()
        
        # 初始化模型
        model = self._setup_model()
        
        # 添加回调函数
        model.add_callback("on_train_epoch_end", self._log_metrics)
        
        print("\n开始训练水下目标检测模型...")
        results = model.train(**self.config)
        
        return model, results

if __name__ == "__main__":
    # 训练配置（针对水下生物优化）
    train_config = {
        'data': r"F:\HW-Underwater_detect_09\src\underwater.yaml",
        'epochs': 100,
        'batch': 16,
        'imgsz': 640,
        'device': '0',
        'optimizer': 'AdamW',
        'lr0': 0.01,
        'cos_lr': True,
        'patience': 20,
        'save_period': 10,
        'project': 'underwater_detection',
        'name': 'exp1',
        'exist_ok': True,
        'seed': 42,
        'pretrained': False,
        'workers': 4,
        'box': 7.5,  # 调整box损失权重
        'cls': 1.5,  # 加强分类损失
        'dfl': 1.5,
        'verbose': True
    }
    
    # 创建训练器并开始训练
    trainer = UnderwaterDetectorTrainer(train_config)
    model, results = trainer.train()
    
    # 输出最佳模型路径
    print(f"\n训练完成！最佳模型保存在:")
    print(f"{results.save_dir}\\weights\\best.pt")