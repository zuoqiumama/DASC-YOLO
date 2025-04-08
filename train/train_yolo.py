import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
from ultralytics import YOLO
import torch
import yaml
from pathlib import Path
import argparse
from datetime import datetime


def train_leather_defect_model(data, device, model_path):
    # 确保CUDA可用
    if not torch.cuda.is_available() and 'cuda' in device:
        raise RuntimeError("CUDA is not available on this system.")
    print(f"Using device: {device}")
    
    # 设置默认CUDA设备
    if 'cuda' in device:
        cuda_id = int(device.split(':')[1])
        # 强制设置当前进程的默认CUDA设备
        torch.cuda.set_device(cuda_id)
        # 只清空指定设备的缓存
        with torch.cuda.device(cuda_id):
            torch.cuda.empty_cache()
        print(f"Set CUDA device to: {cuda_id}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Available memory on {device}: {torch.cuda.get_device_properties(cuda_id).total_memory / 1024**3:.2f} GB")
        print(f"Current memory usage on {device}: {torch.cuda.memory_allocated(cuda_id) / 1024**3:.2f} GB")
    
    # 设置环境变量以避免内存碎片化
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 加载配置文件
    data_yaml_path = Path(f'/home/cigit/project/defect_detect_model_2/{data}/data.yaml')
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    

    model = YOLO(model_path)
    
    # 获取当前时间用于命名
    current_time = datetime.now()
    run_name = f'{model_path.split("/")[-1].split(".")[0]}_{data}_{current_time.strftime("%m%d_%H")}h'
    
    # 高级训练参数配置
    train_args = {
        'data': str(data_yaml_path),
        'epochs': 300,
        'imgsz': 2048,
        'device': device,
        
        # 内存优化设置
        'batch': 8,  # 减小批次大小
        'cache': True,  # 禁用缓存以减少内存使用
        'workers': 4,    # 减少工作线程数
        
        # 指定的数据增强
        'augment': True,
        'flipud': 0.5,    # 上下翻转概率
        'fliplr': 0.5,    # 左右翻转概率
        'hsv_h': 0.0,     # 不改变色调
        'hsv_s': 0.0,     # 不改变饱和度
        'hsv_v': 0.1,     # 仅允许小幅度亮度变化
        
        # 数据增强设置
        'mosaic': 0.0,        # 适当开启马赛克增强
        'mixup': 0.0,         # 适当开启混合增强
        'scale': 0.0,         # 允许小幅度缩放
        'degrees': 0.0,      # 允许小角度旋转
        'translate': 0.0,     # 允许小幅度平移
        
        # 禁用其他增强
        'copy_paste': 0.0,
        'shear': 0.0,
        'perspective': 0.0,
        
        # 优化器设置
        'optimizer': 'AdamW',    # 使用AdamW优化器
        'lr0': 0.001,          # 初始学习率，调整为更小的值
        'lrf': 0.01,           # 最终学习率比例，调整为更合理的衰减比例
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.05,  # 降低预热阶段的学习率
        
        # 损失函数权重
        'box': 7.5,             # 边界框损失权重
        'cls': 1.0,             # 分类损失权重
        'dfl': 1.5,             # DFL损失权重
        
        # 保存设置
        'save': True,
        'save_period': -1,
        'project': 'runs/train',
        'name': run_name,
        
        # 早停设置
        'patience': 100,           # 设置较大的patience值，让模型训练更久
        
        # 额外的性能优化
        'rect': True,           # 使用矩形训练
        
    }
    
    # 开始训练
    try:
        results = model.train(**train_args)
        print("Training completed successfully!")
        
        # 验证模型
        print("Starting validation...")
        val_results = model.val()
        print("Validation completed!")
        
        # 打印关键指标
        print("\nTraining Results:")
        print(f"mAP50: {val_results.box.map50:.4f}")
        print(f"mAP50-95: {val_results.box.map:.4f}")
        
        # 添加测试集评估
        print("\nStarting evaluation on test set...")
        # 从配置文件中提取测试集路径
        test_path = data_config.get('test')
        if test_path:
            # 执行测试集评估
            test_results = model.val(data=str(data_yaml_path), split='test')
            
            # 打印测试集评估结果
            print("\nTest Set Results:")
            print(f"mAP50: {test_results.box.map50:.4f}")
            print(f"mAP50-95: {test_results.box.map:.4f}")
            print(f"Precision: {test_results.box.p:.4f}")
            print(f"Recall: {test_results.box.r:.4f}")
        else:
            print("Test set path not found in the dataset configuration. Skipping test evaluation.")
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLO model for leather defect detection')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., white_dot, black_dot)')
    parser.add_argument('--model', type=str, required=True, help='Dataset name (e.g., white_dot, black_dot)')
    # 改torch可见device，命令行输入默认为0
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on (e.g., cuda:0, cuda:1, cpu)')
    args = parser.parse_args()
    model_path = f'/home/cigit/project/defect_detect_model_2/{args.model}.pt'
    train_leather_defect_model(args.dataset, args.device, model_path) 