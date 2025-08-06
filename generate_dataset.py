"""数据集生成器 - 生成YOLO格式的车标检测数据集"""

import random
import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import List, Tuple, Optional

from config import (
    DATA_CONFIG, DATASET_CONFIG, CLASS_CONFIG, 
    ensure_directories
)
from utils import (
    setup_logging, validate_directory, get_image_files, 
    get_logo_files, load_image, load_logo, rotate_image,
    check_overlap, overlay_image_with_alpha, calculate_yolo_bbox,
    save_yolo_labels, print_progress
)

# 设置日志
logger = setup_logging(__name__)

class DatasetGenerator:
    """数据集生成器类"""
    
    def __init__(self):
        self.background_dir = DATA_CONFIG['background_dir']
        self.logo_dir = DATA_CONFIG['logo_dir']
        self.dataset_dir = DATA_CONFIG['dataset_dir']
        
        # 确保目录存在
        ensure_directories()
        
        # 验证输入目录
        self._validate_input_directories()
        
        # 获取文件列表
        self.background_files = get_image_files(self.background_dir)
        self.logo_files = get_logo_files(self.logo_dir)
        
        # 验证文件
        self._validate_files()
        
        logger.info(f"找到 {len(self.background_files)} 张背景图片")
        logger.info(f"找到 {len(self.logo_files)} 个车标文件")
    
    def _validate_input_directories(self) -> None:
        """验证输入目录"""
        if not validate_directory(self.background_dir, "背景图片"):
            raise FileNotFoundError(f"背景图片目录不存在: {self.background_dir}")
        
        if not validate_directory(self.logo_dir, "车标图片"):
            raise FileNotFoundError(f"车标图片目录不存在: {self.logo_dir}")
    
    def _validate_files(self) -> None:
        """验证文件列表"""
        if not self.background_files:
            raise ValueError(
                f"背景图片目录为空: {self.background_dir}\n"
                "请添加背景图片到此目录后再运行脚本。"
            )
        
        if not self.logo_files:
            raise ValueError(
                f"车标图片目录为空或不包含支持的格式: {self.logo_dir}\n"
                "支持的格式: .png, .svg\n"
                "如果有SVG文件但被跳过，请安装cairosvg: pip install cairosvg"
            )
    
    def _apply_augmentations(self, logo_img: np.ndarray, bg_w: int) -> Optional[np.ndarray]:
        """对车标图像应用数据增强"""
        # 1. 缩放
        scale_min, scale_max = DATASET_CONFIG['logo_scale_range']
        scale = random.uniform(scale_min, scale_max)
        new_w = int(bg_w * scale)
        aspect_ratio = logo_img.shape[0] / logo_img.shape[1]
        new_h = int(new_w * aspect_ratio)
        
        if new_w == 0 or new_h == 0:
            return None
        
        scaled_logo = cv2.resize(logo_img, (new_w, new_h))
        
        # 2. 旋转
        angle_min, angle_max = DATASET_CONFIG['rotation_range']
        angle = random.uniform(angle_min, angle_max)
        rotated_logo = rotate_image(scaled_logo, angle)
        
        return rotated_logo
    
    def _find_placement(self, logo_w: int, logo_h: int, bg_w: int, bg_h: int, 
                       placed_boxes: List[List[int]]) -> Optional[Tuple[int, int]]:
        """寻找车标的合适放置位置"""
        max_attempts = DATASET_CONFIG['placement_attempts']
        
        for _ in range(max_attempts):
            max_x = bg_w - logo_w
            max_y = bg_h - logo_h
            
            if max_x <= 0 or max_y <= 0:
                return None
            
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            
            # 检查重叠
            new_box = [x, y, x + logo_w, y + logo_h]
            is_overlapping = any(check_overlap(new_box, pb) for pb in placed_boxes)
            
            if not is_overlapping:
                return x, y
        
        return None
    
    def generate_augmented_image(self, background_path: Path, output_image_path: Path, 
                               output_label_path: Path) -> bool:
        """生成增强的图像和对应的YOLO标签"""
        try:
            bg = load_image(background_path)
            if bg is None:
                logger.warning(f"无法读取背景图片: {background_path}")
                return False

            bg_h, bg_w = bg.shape[:2]
            labels = []
            placed_boxes = []

            num_logos = random.randint(1, DATASET_CONFIG['max_logos_per_image'])

            for _ in range(num_logos):
                logo_path = random.choice(self.logo_files)
                
                logo_img = load_logo(logo_path)
                if logo_img is None:
                    continue

                # 应用数据增强
                augmented_logo = self._apply_augmentations(logo_img, bg_w)
                if augmented_logo is None:
                    continue
                
                logo_h, logo_w = augmented_logo.shape[:2]

                # 寻找放置位置
                placement = self._find_placement(logo_w, logo_h, bg_w, bg_h, placed_boxes)
                if placement is None:
                    continue
                
                x, y = placement
                placed_boxes.append([x, y, x + logo_w, y + logo_h])
                
                # 叠加图像
                bg = overlay_image_with_alpha(bg, augmented_logo, x, y)

                # 生成YOLO标签 (carlogo类别ID为80)
                x_center, y_center, width, height = calculate_yolo_bbox(
                    x, y, logo_w, logo_h, bg_w, bg_h
                )
                labels.append(f'80 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}')

            if labels:
                cv2.imwrite(str(output_image_path), bg)
                return save_yolo_labels(labels, output_label_path)
            
            return False

        except Exception as e:
            logger.error(f"生成增强图像时发生错误 {background_path}: {e}")
            return False
    
    def create_dataset_split(self, num_images: int, split_name: str) -> int:
        """创建数据集分割（训练集或验证集）"""
        image_dir = self.dataset_dir / split_name / 'images'
        label_dir = self.dataset_dir / split_name / 'labels'
        
        # 确保目录存在
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        generated_count = 0
        
        logger.info(f"开始生成 {num_images} 张 {split_name} 图片...")
        
        for i in range(num_images):
            bg_path = random.choice(self.background_files)
            img_path = image_dir / f'{split_name}_{i}.jpg'
            lbl_path = label_dir / f'{split_name}_{i}.txt'
            
            if self.generate_augmented_image(bg_path, img_path, lbl_path):
                generated_count += 1
            
            # 显示进度
            if (i + 1) % 100 == 0 or i == num_images - 1:
                print_progress(i + 1, num_images, 
                             prefix=f'{split_name}:', 
                             suffix=f'完成 {generated_count}/{i+1}')
        
        logger.info(f"成功生成 {generated_count} 张 {split_name} 图片")
        return generated_count
    
    def generate_dataset(self, num_train: int = 1000, num_val: int = 200) -> None:
        """生成完整的数据集
        
        Args:
            num_train: 训练集图片数量
            num_val: 验证集图片数量
        """
        logger.info("开始生成数据集...")
        logger.info(f"训练集数量: {num_train}")
        logger.info(f"验证集数量: {num_val}")
        
        # 生成训练集
        train_count = self.create_dataset_split(num_train, 'train')
        
        # 生成验证集
        val_count = self.create_dataset_split(num_val, 'val')
        
        # 创建data.yaml配置文件
        self._create_data_yaml()
        
        logger.info(f"数据集生成完成！")
        logger.info(f"训练集: {train_count} 张图片")
        logger.info(f"验证集: {val_count} 张图片")
        logger.info(f"配置文件: {self.dataset_dir / 'data.yaml'}")
    
    def _create_data_yaml(self) -> None:
        """创建YOLO数据集配置文件"""
        yaml_data = {
            'path': str(self.dataset_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'names': CLASS_CONFIG['names']
        }
        
        yaml_path = self.dataset_dir / 'data.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"配置文件已保存: {yaml_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="生成YOLO格式的车标检测数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  生成默认数据集: python generate_dataset.py
  自定义数量: python generate_dataset.py --num_train 2000 --num_val 400
        """
    )
    
    parser.add_argument(
        "--num_train",
        type=int,
        default=1000,
        help="训练集图片数量 (默认: 1000)"
    )
    
    parser.add_argument(
        "--num_val",
        type=int,
        default=200,
        help="验证集图片数量 (默认: 200)"
    )
    
    args = parser.parse_args()
    
    try:
        generator = DatasetGenerator()
        generator.generate_dataset(num_train=args.num_train, num_val=args.num_val)
    except Exception as e:
        logger.error(f"数据集生成失败: {e}")
        raise


if __name__ == '__main__':
    main()