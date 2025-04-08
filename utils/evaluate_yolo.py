import os
import yaml
from ultralytics import YOLO
from pathlib import Path



def evaluate_yolo_model(model_path, dataset_path):
    """评估YOLO模型的性能"""
    # 创建数据配置文件
    data_yaml_path = dataset_path + "/data.yaml"
    
    # 加载预训练模型
    model = YOLO(model_path)
    
    # 在测试集上评估模型
    results = model.val(data=data_yaml_path, split='test')
    
    # 显示评估结果
    print("\n===== 模型评估结果 =====")
    print(f"mAP@0.5: {results.box.map50:.4f}")  # mAP@0.5
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")  # mAP@0.5:0.95
    print(f"精确率 (Precision): {results.box.mp:.4f}")  # 平均精确率
    print(f"召回率 (Recall): {results.box.mr:.4f}")  # 平均召回率
    print(f"F1-Score: {2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr + 1e-16):.4f}")  # F1分数
    
    # 绘制混淆矩阵
    try:
        model.val(data=data_yaml_path, split='test', conf=0.25, plots=True)
        print("\n混淆矩阵和结果图已保存到 'runs/val' 目录")
    except Exception as e:
        print(f"绘制图表时出错: {e}")
    
    return results


def main():
    # 设置路径
    dataset_path = '/home/cigit/project/defect_detect_model_2/white_dot'  # 数据集路径
    model_path = '/home/cigit/project/defect_detect_model_2/pts12/white_680.pt'  # 替换为您的模型路径
    
    # 运行评估
    evaluate_yolo_model(model_path, dataset_path)


if __name__ == '__main__':
    main() 