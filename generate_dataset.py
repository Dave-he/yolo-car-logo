import os
import random
import cv2
from PIL import Image
import numpy as np
import yaml

# Paths
background_dir = '/Users/hyx/codespace/yolo-car-logo/data/no_car/'
logo_dir = '/Users/hyx/codespace/yolo-car-logo/data/car_logo/'
dataset_dir = '/Users/hyx/codespace/yolo-car-logo/dataset/'
train_images_dir = os.path.join(dataset_dir, 'train/images')
train_labels_dir = os.path.join(dataset_dir, 'train/labels')
val_images_dir = os.path.join(dataset_dir, 'val/images')
val_labels_dir = os.path.join(dataset_dir, 'val/labels')

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Class: only one class 'car_logo' which is class 0
classes = {'car_logo': 0}

# Get list of files
backgrounds = [f for f in os.listdir(background_dir) if f.endswith(('.jpg', '.png'))]
logos = [f for f in os.listdir(logo_dir) if f.endswith(('.png', '.svg'))]

# Function to overlay logo on background
def overlay_logo(background_path, logo_path, output_image_path, output_label_path):
    bg = cv2.imread(background_path)
    if bg is None:
        return False
    
    # Load logo, handle SVG if needed (convert to PNG)
    if logo_path.endswith('.svg'):
        # For simplicity, assume SVG can be opened as image, or use library like cairosvg
        # Here using PIL to open, but SVG needs conversion
        try:
            from cairosvg import svg2png
            svg2png(url=logo_path, write_to='temp.png')
            logo = cv2.imread('temp.png', cv2.IMREAD_UNCHANGED)
        except:
            logo = None
    else:
        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    
    if logo is None:
        return False
    
    # Resize logo randomly
    scale = random.uniform(0.1, 0.5)
    logo_h, logo_w = int(logo.shape[0] * scale), int(logo.shape[1] * scale)
    logo = cv2.resize(logo, (logo_w, logo_h))
    
    # Random position
    bg_h, bg_w = bg.shape[:2]
    x = random.randint(0, bg_w - logo_w)
    y = random.randint(0, bg_h - logo_h)
    
    # Overlay (assuming logo has alpha)
    if logo.shape[2] == 4:
        alpha = logo[:, :, 3] / 255.0
        for c in range(3):
            bg[y:y+logo_h, x:x+logo_w, c] = (1 - alpha) * bg[y:y+logo_h, x:x+logo_w, c] + alpha * logo[:, :, c]
    else:
        bg[y:y+logo_h, x:x+logo_w] = logo
    
    cv2.imwrite(output_image_path, bg)
    
    # YOLO label: class x_center y_center width height (normalized)
    x_center = (x + logo_w / 2) / bg_w
    y_center = (y + logo_h / 2) / bg_h
    width = logo_w / bg_w
    height = logo_h / bg_h
    with open(output_label_path, 'w') as f:
        f.write(f'0 {x_center} {y_center} {width} {height}\n')
    return True

# Generate dataset, say 100 for train, 20 for val
num_train = 100
num_val = 20
for i in range(num_train):
    bg = random.choice(backgrounds)
    logo = random.choice(logos)
    overlay_logo(os.path.join(background_dir, bg), os.path.join(logo_dir, logo),
                 os.path.join(train_images_dir, f'train_{i}.jpg'),
                 os.path.join(train_labels_dir, f'train_{i}.txt'))

for i in range(num_val):
    bg = random.choice(backgrounds)
    logo = random.choice(logos)
    overlay_logo(os.path.join(background_dir, bg), os.path.join(logo_dir, logo),
                 os.path.join(val_images_dir, f'val_{i}.jpg'),
                 os.path.join(val_labels_dir, f'val_{i}.txt'))

# Create data.yaml
yaml_data = {
    'path': dataset_dir,
    'train': 'train/images',
    'val': 'val/images',
    'names': {0: 'car_logo'}
}
with open(os.path.join(dataset_dir, 'data.yaml'), 'w') as f:
    yaml.dump(yaml_data, f)
print('Dataset generated.')