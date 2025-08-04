"""工具类 - 包含图像处理、数据验证和其他通用功能"""

import os
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from config import LOG_CONFIG, DATASET_CONFIG

# 检查可选依赖
try:
    from cairosvg import svg2png
    CAIROSVG_AVAILABLE = True
except ImportError:
    CAIROSVG_AVAILABLE = False

def setup_logging(name: str = __name__) -> logging.Logger:
    """设置日志记录"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_CONFIG['log_level']))
    
    if not logger.handlers:
        # 确保日志目录存在
        LOG_CONFIG['log_dir'].mkdir(parents=True, exist_ok=True)
        
        # 创建文件处理器
        log_file = LOG_CONFIG['log_dir'] / f'{name}.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式器
        formatter = logging.Formatter(LOG_CONFIG['log_format'])
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

def validate_directory(directory: Path, name: str, should_exist: bool = True) -> bool:
    """验证目录是否存在且符合要求"""
    logger = setup_logging()
    
    if should_exist:
        if not directory.exists():
            logger.error(f"{name}目录不存在: {directory}")
            return False
        if not directory.is_dir():
            logger.error(f"{name}路径不是目录: {directory}")
            return False
    else:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"创建{name}目录: {directory}")
    
    return True

def validate_file(file_path: Path, description: str = "文件") -> bool:
    """验证文件是否存在"""
    logger = setup_logging()
    
    if not file_path.exists():
        logger.error(f"{description}不存在: {file_path}")
        return False
    
    if not file_path.is_file():
        logger.error(f"{description}不是一个文件: {file_path}")
        return False
    
    return True

def get_image_files(directory: Path, formats: Tuple[str, ...] = None) -> List[Path]:
    """获取目录中的图像文件列表"""
    if formats is None:
        formats = DATASET_CONFIG['supported_image_formats']
    
    image_files = []
    for format_ext in formats:
        image_files.extend(directory.glob(f'*{format_ext}'))
        image_files.extend(directory.glob(f'*{format_ext.upper()}'))
    
    return sorted(image_files)

def get_logo_files(directory: Path) -> List[Path]:
    """获取目录中的车标文件列表"""
    logo_files = []
    
    # PNG文件
    for ext in ['.png', '.PNG']:
        logo_files.extend(directory.glob(f'*{ext}'))
    
    # SVG文件（如果支持）
    if CAIROSVG_AVAILABLE:
        for ext in ['.svg', '.SVG']:
            logo_files.extend(directory.glob(f'*{ext}'))
    
    return sorted(logo_files)

def load_image(image_path: Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """安全地加载图像"""
    logger = setup_logging()
    
    try:
        if not image_path.exists():
            logger.warning(f"图像文件不存在: {image_path}")
            return None
        
        image = cv2.imread(str(image_path), flags)
        if image is None:
            logger.warning(f"无法读取图像文件: {image_path}")
            return None
        
        return image
    except Exception as e:
        logger.error(f"加载图像时发生错误 {image_path}: {e}")
        return None

def load_logo(logo_path: Path) -> Optional[np.ndarray]:
    """加载车标图像，支持PNG和SVG格式"""
    logger = setup_logging()
    
    try:
        if logo_path.suffix.lower() == '.svg':
            if not CAIROSVG_AVAILABLE:
                logger.warning(f"cairosvg未安装，跳过SVG文件: {logo_path}")
                return None
            
            # 转换SVG到PNG
            temp_png_path = logo_path.parent / f'temp_{logo_path.stem}.png'
            svg2png(
                url=str(logo_path),
                write_to=str(temp_png_path),
                output_width=DATASET_CONFIG['svg_output_width']
            )
            
            # 读取临时PNG文件
            logo_img = cv2.imread(str(temp_png_path), cv2.IMREAD_UNCHANGED)
            
            # 删除临时文件
            temp_png_path.unlink(missing_ok=True)
            
            return logo_img
        else:
            return cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)
    
    except Exception as e:
        logger.error(f"加载车标图像时发生错误 {logo_path}: {e}")
        return None

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """旋转图像，扩展图像尺寸以适应旋转后的内容"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 计算新的边界框
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # 调整旋转矩阵以考虑平移
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # 执行旋转
    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(0, 0, 0, 0))
    return rotated

def check_overlap(box1: List[int], box2: List[int]) -> bool:
    """检查两个边界框是否重叠"""
    x1_left, y1_top, x1_right, y1_bottom = box1
    x2_left, y2_top, x2_right, y2_bottom = box2
    
    x_left = max(x1_left, x2_left)
    y_top = max(y1_top, y2_top)
    x_right = min(x1_right, x2_right)
    y_bottom = min(y1_bottom, y2_bottom)
    
    return x_right > x_left and y_bottom > y_top

def overlay_image_with_alpha(background: np.ndarray, overlay: np.ndarray, 
                            x: int, y: int) -> np.ndarray:
    """在背景图像上叠加带有透明度的图像"""
    h, w = overlay.shape[:2]
    
    if overlay.shape[2] == 4:  # 有alpha通道
        alpha = overlay[:, :, 3] / 255.0
        alpha = alpha[:, :, np.newaxis]
        
        # 混合
        background[y:y+h, x:x+w] = \
            background[y:y+h, x:x+w] * (1 - alpha) + \
            overlay[:, :, :3] * alpha
    else:
        background[y:y+h, x:x+w] = overlay[:, :, :3]
    
    return background

def calculate_yolo_bbox(x: int, y: int, w: int, h: int, 
                       img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """计算YOLO格式的边界框坐标"""
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    width = w / img_w
    height = h / img_h
    
    return x_center, y_center, width, height

def save_yolo_labels(labels: List[str], label_path: Path) -> bool:
    """保存YOLO格式的标签文件"""
    logger = setup_logging()
    
    try:
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(labels) + '\n')
        return True
    except Exception as e:
        logger.error(f"保存标签文件时发生错误 {label_path}: {e}")
        return False

def print_progress(current: int, total: int, prefix: str = '', suffix: str = '', 
                  length: int = 50) -> None:
    """打印进度条"""
    percent = (current / total) * 100
    filled_length = int(length * current // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='', flush=True)
    if current == total:
        print()  # 换行