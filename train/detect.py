from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import numpy as np
import time

def detect_defects(model_path, image_folder, output_folder, conf_threshold=0.25):
    """
    对指定文件夹中的图片进行缺陷检测
    :param model_path: 训练好的模型路径
    :param image_folder: 待检测图片文件夹路径
    :param output_folder: 结果输出文件夹路径
    :param conf_threshold: 置信度阈值
    """
    # 加载模型
    model = YOLO(model_path)
    
    # 创建输出文件夹
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # 创建结果文本文件
    results_file = output_folder / 'detection_results.txt'
    
    # 获取所有支持的图片文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in Path(image_folder).glob('*') if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images to process")
    
    # 初始化时间统计
    total_time = 0
    
    with open(results_file, 'w', encoding='utf-8') as f:
        for image_path in image_files:
            print(f"\nProcessing {image_path.name}")
            
            # 开始计时
            start_time = time.time()
            
            # 进行检测
            results = model.predict(
                source=str(image_path),
                conf=conf_threshold,
                save=False,
                save_txt=False
            )[0]
            
            # 获取原始图片
            image = cv2.imread(str(image_path))
            
            # 写入检测结果到文件
            f.write(f"\n=== {image_path.name} ===\n")
            
            if len(results.boxes) == 0:
                f.write("No defects detected\n")
                
            # 遍历所有检测到的目标
            for i, box in enumerate(results.boxes):
                # 获取坐标和置信度
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = results.names[class_id]
                
                # 写入结果
                f.write(f"Defect {i+1}:\n")
                f.write(f"  Type: {class_name}\n")
                f.write(f"  Confidence: {confidence:.2f}\n")
                f.write(f"  Coordinates: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})\n")
                
                # 在图片上绘制边界框
                cv2.rectangle(
                    image,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )
                
                # 添加标签文本
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(
                    image,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
            
            # 保存标注后的图片
            output_image_path = output_folder / f"detected_{image_path.name}"
            cv2.imwrite(str(output_image_path), image)
            
            # 结束计时并累加
            end_time = time.time()
            process_time = end_time - start_time
            total_time += process_time
            
            print(f"Processed {image_path.name} in {process_time:.3f} seconds")
            print(f"Saved annotated image to {output_image_path}")
        
        # 计算并输出平均处理时间
        avg_time = total_time / len(image_files)
        print(f"\nAverage processing time per image: {avg_time:.3f} seconds")
        f.write(f"\n=== Performance Statistics ===\n")
        f.write(f"Total images processed: {len(image_files)}\n")
        f.write(f"Total processing time: {total_time:.3f} seconds\n")
        f.write(f"Average processing time per image: {avg_time:.3f} seconds\n")

if __name__ == '__main__':
    data = ""
    conf = 0.25
    # 设置路径
    model_path = "/home/cigit/project/defect_detect_model_2/runs/train/yolov5n6u_all_0404_09h/weights/best.pt"  # 使用训练好的最佳模型
    image_folder = f"/home/cigit/project/defect_detect_model/cut"  # 待检测的图片文件夹
    output_folder = f"/home/cigit/project/defect_detect_model/cut/5n"  # 检测结果输出文件夹
    
    # 执行检测
    detect_defects(model_path, image_folder, output_folder, conf_threshold=conf) 