# YOLO车标检测项目

一个基于YOLOv8的车标检测项目，支持数据集生成、模型训练和预测功能。

## 项目结构

```
yolo-car-logo/
├── config.py              # 项目配置文件
├── utils.py               # 工具函数
├── generate_dataset.py    # 数据集生成脚本
├── main.py                # 主程序（训练和预测）
├── requirements.txt       # 项目依赖
├── README.md              # 项目说明
├── data/                  # 数据目录
│   ├── no_car/           # 无车图片（背景图）
│   ├── car_logo/         # 车标图片
│   └── has_car/          # 待预测图片
├── dataset/              # 生成的YOLO数据集
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml
└── runs/                 # 训练和预测结果
    ├── train/
    └── predict/
```

## 功能特性

### 数据集生成
- 支持多种图片格式（JPG, PNG, JPEG, BMP, TIFF）
- 支持SVG车标自动转换
- 车标随机旋转（-30°到+30°）
- 车标随机缩放（0.05到0.3倍）
- 每张背景图可放置1-4个车标
- 智能防重叠算法
- 混合车标类型支持
- 自动生成YOLO格式标注

### 模型训练
- 基于YOLOv8架构
- 支持自定义训练参数
- 自动保存最佳模型
- 详细的训练日志

### 模型预测
- 支持批量图片预测
- 自动查找最新训练模型
- 可视化预测结果
- 可调节置信度和IoU阈值

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 准备数据

将数据放入对应目录：
- `data/no_car/`: 放入无车的背景图片
- `data/car_logo/`: 放入车标图片（支持SVG格式）
- `data/has_car/`: 放入需要预测的图片

### 2. 生成数据集

```bash
python generate_dataset.py
```

这将生成：
- 1000张训练图片
- 200张验证图片
- 对应的YOLO格式标注文件
- `dataset/data.yaml`配置文件

### 3. 训练模型

```bash
# 基础训练
python main.py train

# 自定义参数训练
python main.py train --epochs 100 --imgsz 640
```

### 4. 预测图片

```bash
# 使用最新训练的模型预测
python main.py predict --model auto

# 使用指定模型预测
python main.py predict --model runs/train/exp/weights/best.pt

# 指定预测目录
python main.py predict --source data/has_car
```

## 配置说明

所有配置参数都在`config.py`中定义，包括：

- **数据集配置**: 训练/验证集数量、图片格式、车标参数等
- **训练配置**: 模型名称、训练轮数、批次大小等
- **预测配置**: 置信度阈值、IoU阈值、输出目录等
- **日志配置**: 日志级别和格式

## 命令行参数

### 训练参数
- `--data`: 数据集YAML文件路径
- `--model`: 预训练模型名称
- `--epochs`: 训练轮数
- `--imgsz`: 图片尺寸

### 预测参数
- `--model`: 模型文件路径（使用'auto'自动查找最新模型）
- `--source`: 预测图片目录
- `--conf`: 置信度阈值
- `--iou`: IoU阈值

## 示例用法

```bash
# 生成数据集
python generate_dataset.py

# 训练模型（100轮）
python main.py train --epochs 100

# 使用最新模型预测
python main.py predict --model auto --source data/has_car

# 使用指定模型和参数预测
python main.py predict --model runs/train/exp/weights/best.pt --conf 0.5 --iou 0.4
```

## 注意事项

1. 确保`data/no_car/`和`data/car_logo/`目录中有足够的图片
2. SVG文件会自动转换为PNG格式
3. 训练结果保存在`runs/train/`目录
4. 预测结果保存在`runs/predict/`目录
5. 建议使用GPU进行训练以提高速度

## 故障排除

### 常见问题

1. **ImportError**: 确保已安装所有依赖包
2. **FileNotFoundError**: 检查数据目录是否存在且包含图片
3. **CUDA错误**: 如果没有GPU，模型会自动使用CPU
4. **内存不足**: 减少批次大小或图片尺寸

### 日志查看

程序运行时会输出详细日志，包括：
- 数据集生成进度
- 训练过程信息
- 预测结果统计
- 错误和警告信息

## 许可证

本项目仅供学习和研究使用。