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
from tqdm import tqdm

logger = setup_logging(__name__)

# 配置
ANNOTATION_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
IMAGE_URL = 'http://images.cocodataset.org/zips/val2017.zip'
TARGET_DIR = PROJECT_ROOT / 'data' / 'no_car'
MAX_IMAGES = 100  # 最大下载图片数，防止过多


def download_and_extract(url: str, target_dir: str, filename: str = None):
    """下载并解压ZIP文件，显示进度条并检查已存在文件"""
    if filename is None:
        filename = url.split('/')[-1]
    
    # 检查文件是否已存在
    if os.path.exists(filename):
        logger.info(f"文件已存在，跳过下载: {filename}")
    else:
        logger.info(f"下载 {url}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    # 解压文件
    logger.info(f"解压 {filename} 到 {target_dir}...")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    logger.info(f"解压完成到 {target_dir}")


def main():
    
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    # 下载并解压注解
    ann_dir = PROJECT_ROOT / 'annotations'
    ann_filename = 'annotations_trainval2017.zip'
    download_and_extract(ANNOTATION_URL, PROJECT_ROOT, ann_filename)
    
    # 加载COCO注解
    coco = COCO(str(ann_dir / 'instances_val2017.json'))
    
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
    img_filename = 'val2017.zip'
    download_and_extract(IMAGE_URL, PROJECT_ROOT, img_filename)
    
    # 复制选中的图片到目标目录
    images = coco.loadImgs(no_car_img_ids)
    with tqdm(total=len(images), desc="复制图片") as pbar:
        for img in images:
            src_path = img_dir / img['file_name']
            dst_path = TARGET_DIR / img['file_name']
            if src_path.exists():
                os.rename(src_path, dst_path)  # 移动以节省空间
                logger.info(f"移动 {img['file_name']} 到 {TARGET_DIR}")
            else:
                logger.warning(f"图片不存在: {src_path}")
            pbar.update(1)
    
    # 清理临时目录
    import shutil
    shutil.rmtree(img_dir)
    shutil.rmtree(ann_dir)
    
    logger.info(f"下载完成！图片保存在 {TARGET_DIR}")

if __name__ == "__main__":
    main()