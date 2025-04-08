import os
import json
import shutil
from pathlib import Path
import cv2
import numpy as np

def convert_labelme_to_yolo(labelme_path, output_path):
    # 创建输出目录结构
    output_path = Path(output_path)
    (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # 获取所有类别
    classes = set()
    json_files = list(Path(labelme_path).glob('*.json'))
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for shape in data['shapes']:
                classes.add(shape['label'])
    print(f"get classes {classes}")
    
    classes = sorted(list(classes))
    class_dict = {class_name: i for i, class_name in enumerate(classes)}
    
    # 保存类别文件
    with open(output_path / 'classes.txt', 'w', encoding='utf-8') as f:
        for class_name in classes:
            f.write(f'{class_name}\n')
    
    # 转换标注文件
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取图片尺寸
        img_path = Path(labelme_path) / data['imagePath']
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Cannot read image {img_path}")
            continue
            
        img_height, img_width = img.shape[:2]
        
        # 创建YOLO格式的标注文件
        txt_filename = json_file.stem + '.txt'
        yolo_annotations = []
        
        for shape in data['shapes']:
            # 获取类别ID
            class_id = class_dict[shape['label']]
            
            # 获取多边形点
            points = np.array(shape['points'])
            
            # 计算边界框
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            
            # 转换为YOLO格式（归一化的中心点坐标和宽高）
            x_center = ((x_min + x_max) / 2) / img_width
            y_center = ((y_min + y_max) / 2) / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            
            # 确保值在0-1之间
            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            width = min(max(width, 0.0), 1.0)
            height = min(max(height, 0.0), 1.0)
            
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # 随机分配训练集和验证集（80%训练，20%验证）
        is_train = np.random.random() < 0.80
        subset = 'train' if is_train else 'val'
        
        # 复制图片
        shutil.copy2(img_path, output_path / 'images' / subset / data['imagePath'])
        
        # 保存标注文件
        with open(output_path / 'labels' / subset / txt_filename, 'w') as f:
            f.write('\n'.join(yolo_annotations))

if __name__ == '__main__':
    labelme_path = '/home/cigit/project/defect/white_dot'  # 修改为您的labelme数据集路径
    output_path = '/home/cigit/project/defect_detect_model/white_dot'      # 修改为您想要输出的路径
    convert_labelme_to_yolo(labelme_path, output_path) 