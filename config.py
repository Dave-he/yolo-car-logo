"""配置文件 - 管理项目的所有配置参数"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据路径配置
DATA_CONFIG = {
    'background_dir': PROJECT_ROOT / 'data' / 'no_car',
    'logo_dir': PROJECT_ROOT / 'data' / 'car_logo',
    'prediction_dir': PROJECT_ROOT / 'data' / 'has_car',
    'dataset_dir': PROJECT_ROOT / 'dataset',
}

# 数据集生成配置
DATASET_CONFIG = {
    'num_train': 1000,
    'num_val': 200,
    'supported_image_formats': ('.jpg', '.png', '.jpeg', '.bmp', '.tiff'),
    'supported_logo_formats': ('.png', '.svg'),
    'logo_scale_range': (0.05, 0.15),  # 车标缩放范围（相对于背景宽度的百分比）
    'rotation_range': (-30, 30),  # 旋转角度范围
    'max_logos_per_image': 4,  # 每张图片最多车标数量
    'placement_attempts': 10,  # 放置车标时的最大尝试次数
    'svg_output_width': 256,  # SVG转PNG时的输出宽度
}

# 训练配置
TRAIN_CONFIG = {
    'default_model': 'yolo11n.pt',
    'default_epochs': 50,
    'default_imgsz': 640,
    'project_name': 'runs/train',
    'experiment_name': 'car_logo_exp',
}

# 预测配置
PREDICT_CONFIG = {
    'project_name': 'runs/predict',
    'experiment_name': 'car_logo_exp',
    'confidence_threshold': 0.25,
    'iou_threshold': 0.45,
}

# 日志配置
LOG_CONFIG = {
    'log_dir': PROJECT_ROOT / 'logs',
    'log_level': 'INFO',
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
}

# 类别配置
CLASS_CONFIG = {
    'names': {0: 'car_logo'},
    'num_classes': 1,
}

# 确保必要的目录存在
def ensure_directories():
    """确保所有必要的目录存在"""
    directories = [
        DATA_CONFIG['dataset_dir'],
        DATA_CONFIG['dataset_dir'] / 'train' / 'images',
        DATA_CONFIG['dataset_dir'] / 'train' / 'labels',
        DATA_CONFIG['dataset_dir'] / 'val' / 'images',
        DATA_CONFIG['dataset_dir'] / 'val' / 'labels',
        LOG_CONFIG['log_dir'],
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    ensure_directories()
    print("配置初始化完成，所有必要目录已创建。")