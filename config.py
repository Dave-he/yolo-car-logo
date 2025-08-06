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
    'supported_image_formats': ('.jpg', '.png', '.jpeg', '.bmp', '.tiff'),
    'supported_logo_formats': ('.png', '.svg'),
    'logo_scale_range': (0.05, 0.15),  # 车标缩放范围（相对于背景宽度的百分比）
    'rotation_range': (-30, 30),  # 旋转角度范围
    'max_logos_per_image': 4,  # 每张图片最多车标数量
    'placement_attempts': 10,  # 放置车标时的最大尝试次数
    'svg_output_width': 256,  # SVG转PNG时的输出宽度
    'data_yaml_path': 'dataset/data.yaml'
}

# 训练配置
TRAIN_CONFIG = {
    'project_path': 'runs/train',
    'experiment_name': 'car_logo_exp',
    'model_name': 'yolo11l.pt',
    'epochs': 100,
    'image_size': 640,
    'batch_size': 16
}

# 预测配置
PREDICT_CONFIG = {
    'source_dir' : DATA_CONFIG['prediction_dir'],
    'project_path': 'runs/predict',
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

# 类别配置 - COCO 80个原始类别 + carlogo类别
CLASS_CONFIG = {
    'names': {
        # COCO 80个原始类别 (0-79)
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        4: 'airplane',
        5: 'bus',
        6: 'train',
        7: 'truck',
        8: 'boat',
        9: 'traffic light',
        10: 'fire hydrant',
        11: 'stop sign',
        12: 'parking meter',
        13: 'bench',
        14: 'bird',
        15: 'cat',
        16: 'dog',
        17: 'horse',
        18: 'sheep',
        19: 'cow',
        20: 'elephant',
        21: 'bear',
        22: 'zebra',
        23: 'giraffe',
        24: 'backpack',
        25: 'umbrella',
        26: 'handbag',
        27: 'tie',
        28: 'suitcase',
        29: 'frisbee',
        30: 'skis',
        31: 'snowboard',
        32: 'sports ball',
        33: 'kite',
        34: 'baseball bat',
        35: 'baseball glove',
        36: 'skateboard',
        37: 'surfboard',
        38: 'tennis racket',
        39: 'bottle',
        40: 'wine glass',
        41: 'cup',
        42: 'fork',
        43: 'knife',
        44: 'spoon',
        45: 'bowl',
        46: 'banana',
        47: 'apple',
        48: 'sandwich',
        49: 'orange',
        50: 'broccoli',
        51: 'carrot',
        52: 'hot dog',
        53: 'pizza',
        54: 'donut',
        55: 'cake',
        56: 'chair',
        57: 'couch',
        58: 'potted plant',
        59: 'bed',
        60: 'dining table',
        61: 'toilet',
        62: 'tv',
        63: 'laptop',
        64: 'mouse',
        65: 'remote',
        66: 'keyboard',
        67: 'cell phone',
        68: 'microwave',
        69: 'oven',
        70: 'toaster',
        71: 'sink',
        72: 'refrigerator',
        73: 'book',
        74: 'clock',
        75: 'vase',
        76: 'scissors',
        77: 'teddy bear',
        78: 'hair drier',
        79: 'toothbrush',
        # 新增的carlogo类别 (80)
        80: 'carlogo'
    },
    'num_classes': 81,
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