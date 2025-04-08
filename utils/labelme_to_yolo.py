import os
import json
import shutil
import random
from pathlib import Path
import numpy as np

def convert_labelme_to_yolo(input_dir, output_dir, class_names=None):
    """
    将Labelme格式的标注数据转换为YOLO格式，并分割为训练、验证和测试集
    
    参数:
        input_dir: 输入目录，包含图片和labelme的json文件
        output_dir: 输出目录，用于存储YOLO格式的数据集
        class_names: 类别名称列表，如果为None，将从所有json文件中自动提取
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # 创建输出目录结构
    os.makedirs(output_dir, exist_ok=True)
    for dataset in ['train', 'val']:
        os.makedirs(output_dir / 'images' / dataset, exist_ok=True)
        os.makedirs(output_dir / 'labels' / dataset, exist_ok=True)
    
    # 获取所有图片和对应的json文件
    image_files = []
    json_files = []
    
    # 支持的图片格式
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    for file in os.listdir(input_dir):
        file_path = input_dir / file
        if file.lower().endswith('.json'):
            json_files.append(file_path)
        elif any(file.lower().endswith(ext) for ext in img_extensions):
            image_files.append(file_path)
    
    # 如果没有提供类别名称，自动从所有json文件中提取
    if class_names is None:
        class_names = set()
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for shape in data['shapes']:
                    class_names.add(shape['label'])
        class_names = sorted(list(class_names))
    
    # 创建类别文件
    with open(output_dir / 'classes.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(class_names))
    
    print(f"找到的类别: {class_names}")
    
    # 处理每个json文件
    valid_pairs = []
    for json_file in json_files:
        # 查找对应的图片文件
        image_file = None
        json_name = json_file.stem
        for img_ext in img_extensions:
            img_path = input_dir / f"{json_name}{img_ext}"
            if img_path in image_files:
                image_file = img_path
                break
        
        if image_file is None:
            print(f"警告: 找不到对应的图片文件 {json_name}")
            continue
        
        valid_pairs.append((image_file, json_file))
    
    # 打乱数据集并按照8:1:1分割
    random.shuffle(valid_pairs)
    total = len(valid_pairs)
    train_count = int(total * 0.8)
    val_count = int(total * 0.2)
    
    train_pairs = valid_pairs[:train_count]
    val_pairs = valid_pairs[train_count:train_count + val_count]
    
    
    # 处理每个数据集
    datasets = {
        'train': train_pairs,
        'val': val_pairs,
    }
    
    for dataset_name, pairs in datasets.items():
        dataset_images_dir = output_dir / 'images' / dataset_name
        dataset_labels_dir = output_dir / 'labels' / dataset_name
        
        for image_file, json_file in pairs:
            # 复制图片
            shutil.copy2(image_file, dataset_images_dir / image_file.name)
            
            # 转换并保存标签
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                img_width = data['imageWidth']
                img_height = data['imageHeight']
                
                # 创建YOLO格式的标签文件
                yolo_file = dataset_labels_dir / f"{json_file.stem}.txt"
                with open(yolo_file, 'w', encoding='utf-8') as f:
                    for shape in data['shapes']:
                        # 跳过非多边形标注
                        if shape['shape_type'] != 'polygon' and shape['shape_type'] != 'rectangle':
                            continue
                        
                        label = shape['label']
                        class_id = class_names.index(label)
                        
                        points = shape['points']
                        if shape['shape_type'] == 'rectangle':
                            # 矩形格式转换为YOLO格式
                            # Labelme的矩形标注为左上和右下两个点
                            x1, y1 = points[0]
                            x2, y2 = points[1]
                            
                            # 计算YOLO格式的中心点和宽高
                            x_center = (x1 + x2) / 2 / img_width
                            y_center = (y1 + y2) / 2 / img_height
                            width = abs(x2 - x1) / img_width
                            height = abs(y2 - y1) / img_height
                            
                            # 写入YOLO格式: class_id center_x center_y width height
                            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                        elif shape['shape_type'] == 'polygon':
                            # 多边形转换为边界框
                            x_coords = [p[0] for p in points]
                            y_coords = [p[1] for p in points]
                            
                            x_min, x_max = min(x_coords), max(x_coords)
                            y_min, y_max = min(y_coords), max(y_coords)
                            
                            # 计算YOLO格式的中心点和宽高
                            x_center = (x_min + x_max) / 2 / img_width
                            y_center = (y_min + y_max) / 2 / img_height
                            width = (x_max - x_min) / img_width
                            height = (y_max - y_min) / img_height
                            
                            # 写入YOLO格式: class_id center_x center_y width height
                            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
            except Exception as e:
                print(f"处理 {json_file} 时出错: {e}")
    
    # 创建YOLO配置文件
    create_yolo_config(output_dir, class_names)
    
    print(f"数据集转换完成:")
    print(f"训练集: {len(train_pairs)} 图片")
    print(f"验证集: {len(val_pairs)} 图片")
    print(f"输出目录: {output_dir}")

def create_yolo_config(output_dir, class_names):
    """创建YOLO训练需要的配置文件"""
    output_dir = Path(output_dir)
    
    # 创建data.yaml文件
    data_yaml = {
        'train': './images/train',
        'val': './images/val',
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(output_dir / 'data.yaml', 'w', encoding='utf-8') as f:
        yaml_str = "train: ./images/train\n"
        yaml_str += "val: ./images/val\n"
        yaml_str += f"nc: {len(class_names)}\n"
        yaml_str += "names: ["
        yaml_str += ", ".join([f"'{name}'" for name in class_names])
        yaml_str += "]\n"
        f.write(yaml_str)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='将Labelme标注转换为YOLO格式，并分割数据集')
    parser.add_argument('--input_dir', required=True, help='包含图片和labelme标注的输入目录')
    parser.add_argument('--output_dir', required=True, help='YOLO格式数据集的输出目录')
    parser.add_argument('--classes', default=None, help='可选的类别文件，每行一个类别名称')
    
    args = parser.parse_args()
    
    class_names = None
    if args.classes:
        with open(args.classes, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
    
    convert_labelme_to_yolo(args.input_dir, args.output_dir, class_names) 