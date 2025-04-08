from ultralytics import YOLO
import argparse

def export_to_onnx(model_path, imgsz=2048):
    """
    将YOLOv8模型导出为ONNX格式
    :param model_path: 训练好的模型路径 (.pt文件)
    :param imgsz: 输入图像尺寸
    """
    # 加载模型
    model = YOLO(model_path)
    
    # 导出为ONNX格式
    # opset=12 是为了更好的兼容性
    # dynamic=True 允许动态batch size
    # simplify=True 简化模型结构
    success = model.export(format="onnx", imgsz=imgsz, dynamic=False, opset=16)
    
    if success:
        print(f"Model exported successfully to {model_path.replace('.pt', '.onnx')}")
    else:
        print("Export failed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Ymodel to ONNX format")
    parser.add_argument("--model", type=str, required=True, help="Path to the .pt model file")
    parser.add_argument("--imgsz", type=int, default=2048, help="Input image size")
    args = parser.parse_args()
    
    export_to_onnx(args.model, args.imgsz) 