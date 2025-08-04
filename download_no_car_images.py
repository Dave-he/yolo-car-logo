#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载COCO数据集不包含汽车的图片作为背景
"""

import os
import requests
import zipfile
import io
from pycocotools.coco import COCO
from config import PROJECT_ROOT
from utils import setup_logging

logger = setup_logging(__name__)

# 配置
ANNOTATION_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
IMAGE_URL = 'http://images.cocodataset.org/zips/val2017.zip'
TARGET_DIR = PROJECT_ROOT / 'data' / 'no_car'
MAX_IMAGES = 100  # 最大下载图片数，防止过多


def download_and_extract(url: str, target_dir: str):
    """下载并解压ZIP文件"""
    logger.info(f"下载 {url}...")
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(target_dir)
    logger.info(f"解压完成到 {target_dir}")


def main():
    
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    # 下载并解压注解
    ann_dir = PROJECT_ROOT / 'annotations'
    download_and_extract(ANNOTATION_URL, PROJECT_ROOT)
    
    # 加载COCO注解
    coco = COCO(str(ann_dir / 'annotations' / 'instances_val2017.json'))
    
    # 获取'car'类别ID
    car_cat_id = coco.getCatIds(catNms=['car'])[0]
    logger.info(f"'car' 类别ID: {car_cat_id}")
    
    # 获取所有图片ID
    all_img_ids = coco.getImgIds()
    
    # 过滤不包含汽车的图片
    no_car_img_ids = []
    for img_id in all_img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[car_cat_id], iscrowd=None)
        if not ann_ids:
            no_car_img_ids.append(img_id)
        if len(no_car_img_ids) >= MAX_IMAGES:
            break
    
    logger.info(f"找到 {len(no_car_img_ids)} 张不包含汽车的图片")
    
    # 下载图片
    img_dir = PROJECT_ROOT / 'val2017'
    download_and_extract(IMAGE_URL, PROJECT_ROOT)
    
    # 复制选中的图片到目标目录
    images = coco.loadImgs(no_car_img_ids)
    for img in images:
        src_path = img_dir / img['file_name']
        dst_path = TARGET_DIR / img['file_name']
        if src_path.exists():
            os.rename(src_path, dst_path)  # 移动以节省空间
            logger.info(f"移动 {img['file_name']} 到 {TARGET_DIR}")
        else:
            logger.warning(f"图片不存在: {src_path}")
    
    # 清理临时目录
    import shutil
    shutil.rmtree(img_dir)
    shutil.rmtree(ann_dir)
    
    logger.info(f"下载完成！图片保存在 {TARGET_DIR}")

if __name__ == "__main__":
    main()