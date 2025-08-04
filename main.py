#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO车标检测模型训练和预测主程序
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from config import (
    DATASET_CONFIG, TRAIN_CONFIG, PREDICT_CONFIG,
    ensure_directories
)
from utils import (
    setup_logging, validate_directory, validate_file,
    get_image_files
)

# 设置日志
logger = setup_logging(__name__)


class YOLOTrainer:
    """YOLO模型训练器"""
    
    def __init__(self):
        ensure_directories()
    
    def train(self, model_name: str, data_yaml: Path, epochs: int, 
              imgsz: int, batch: Optional[int] = None) -> bool:
        """
        训练YOLO模型
        
        Args:
            model_name: 预训练模型名称
            data_yaml: 数据集配置文件路径
            epochs: 训练轮数
            imgsz: 图片尺寸
            batch: 批次大小
        
        Returns:
            bool: 训练是否成功
        """
        try:
            # 验证数据集配置文件
            if not validate_file(data_yaml, "数据集配置文件"):
                logger.error("请先运行 generate_dataset.py 生成数据集")
                return False
            
            logger.info(f"开始训练...")
            logger.info(f"模型: {model_name}")
            logger.info(f"数据集: {data_yaml}")
            logger.info(f"训练轮数: {epochs}")
            logger.info(f"图片尺寸: {imgsz}")
            
            # 加载模型
            model = YOLO(model_name)
            
            # 设置训练参数
            train_kwargs = {
                'data': str(data_yaml),
                'epochs': epochs,
                'imgsz': imgsz,
                'project': TRAIN_CONFIG['project_name'],
                'name': TRAIN_CONFIG['experiment_name'],
                'save': True,
                'plots': True,
                'val': True
            }
            
            if batch:
                train_kwargs['batch'] = batch
            
            # 开始训练
            results = model.train(**train_kwargs)
            
            logger.info("训练完成！")
            logger.info(f"结果保存到: {results.save_dir}")
            logger.info(f"最佳模型: {results.save_dir}/weights/best.pt")
            
            return True
            
        except Exception as e:
            logger.error(f"训练过程中发生错误: {e}")
            return False


class YOLOPredictor:
    """YOLO模型预测器"""
    
    def __init__(self):
        ensure_directories()
    
    def _find_latest_model(self) -> Path:
        """查找最新训练的模型"""
        train_dir = Path(TRAIN_CONFIG['project_name'])
        model_pattern = f"*/weights/best.pt"
        
        model_files = list(train_dir.glob(model_pattern))
        if not model_files:
            raise FileNotFoundError("未找到训练好的模型，请先训练模型")
        
        # 按修改时间排序，返回最新的
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"找到最新训练的模型: {latest_model}")
        return latest_model
    
    def predict(self, model_path: str, source_dir: Path) -> bool:
        """
        使用训练好的YOLO模型进行预测
        
        Args:
            model_path: 模型文件路径
            source_dir: 预测图片目录
        
        Returns:
            bool: 预测是否成功
        """
        try:
            # 处理模型路径
            if model_path == 'auto' or not Path(model_path).exists():
                if model_path != 'auto':
                    logger.warning(f"指定的模型文件不存在: {model_path}")
                model_path = self._find_latest_model()
            else:
                model_path = Path(model_path)
            
            # 验证预测目录
            if not validate_directory(source_dir, "预测图片"):
                return False
            
            # 检查目录中是否有图片
            image_files = get_image_files(source_dir)
            if not image_files:
                logger.error(f"预测目录中没有找到图片: {source_dir}")
                logger.info("请添加图片到预测目录中")
                return False
            
            logger.info(f"开始预测...")
            logger.info(f"模型: {model_path}")
            logger.info(f"预测目录: {source_dir}")
            logger.info(f"找到 {len(image_files)} 张图片")
            
            # 加载模型
            model = YOLO(str(model_path))
            
            # 进行预测
            results = model.predict(
                source=str(source_dir),
                save=True,
                project=PREDICT_CONFIG['project_name'],
                name=PREDICT_CONFIG['experiment_name'],
                conf=PREDICT_CONFIG['confidence_threshold'],
                iou=PREDICT_CONFIG['iou_threshold']
            )
            
            logger.info("预测完成！")
            logger.info(f"结果保存到: {PREDICT_CONFIG['project_name']}/{PREDICT_CONFIG['experiment_name']}")
            
            return True
            
        except Exception as e:
            logger.error(f"预测过程中发生错误: {e}")
            return False


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(
        description="YOLO车标检测模型训练和预测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  训练模型: python main.py train --epochs 100
  预测图片: python main.py predict --source data/has_car
  使用自动模型预测: python main.py predict --model auto --source data/has_car
        """
    )
    
    parser.add_argument(
        "action", 
        choices=["train", "predict"], 
        help="执行的操作: train(训练) 或 predict(预测)"
    )
    
    parser.add_argument(
        "--data", 
        default=str(DATASET_CONFIG['data_yaml_path']),
        help="数据集YAML文件路径 (用于训练)"
    )
    
    parser.add_argument(
        "--model", 
        default=TRAIN_CONFIG['model_name'],
        help="模型文件路径 (训练时为预训练模型，预测时为训练好的模型，使用'auto'自动查找最新模型)"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=TRAIN_CONFIG['epochs'],
        help="训练轮数"
    )
    
    parser.add_argument(
        "--source", 
        default=str(PREDICT_CONFIG['source_dir']),
        help="预测图片目录路径"
    )
    
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=TRAIN_CONFIG['image_size'],
        help="图片尺寸"
    )
    
    parser.add_argument(
        "--batch", 
        type=int, 
        default=TRAIN_CONFIG['batch_size'],
        help="批次大小"
    )
    
    parser.add_argument(
        "--conf", 
        type=float, 
        default=PREDICT_CONFIG['confidence_threshold'],
        help="预测置信度阈值"
    )
    
    parser.add_argument(
        "--iou", 
        type=float, 
        default=PREDICT_CONFIG['iou_threshold'],
        help="IoU阈值"
    )
    
    args = parser.parse_args()
    
    try:
        if args.action == "train":
            trainer = YOLOTrainer()
            success = trainer.train(
                model_name=args.model,
                data_yaml=Path(args.data),
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch
            )
            if not success:
                logger.error("训练失败")
                return 1
                
        elif args.action == "predict":
            predictor = YOLOPredictor()
            success = predictor.predict(
                model_path=args.model,
                source_dir=Path(args.source)
            )
            if not success:
                logger.error("预测失败")
                return 1
        
        logger.info("操作完成！")
        return 0
        
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        return 1
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        return 1


if __name__ == "__main__":
    exit(main())