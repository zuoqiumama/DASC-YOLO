import torch
from thop import profile
import time

def cleanup_model(model):
    """清除模型中可能残留的thop计算属性"""
    for m in model.modules():
        for attr in ['total_ops', 'total_params']:
            if hasattr(m, attr):
                delattr(m, attr)

def get_model_metrics(model, input_shape=(3, 200, 200), device='cuda', test_round=100):
    # 确保获取实际模型对象（关键修改）
    torch_model = model.model if hasattr(model, 'model') else model
    torch_model = torch_model.to(device)
    # # 参数数量计算
    # params = sum(p.numel() for p in torch_model.parameters()) / 1e6
    
    # # FLOPs计算前的清理（关键修改）
    # cleanup_model(torch_model)
    
    # # FLOPs计算
    input_tensor = torch.randn(1, *input_shape).to(device)
    # torch_model.to(device)
    # with torch.no_grad():
    #     flops, _ = profile(torch_model, inputs=(input_tensor,), verbose=False)
    # gflops = flops / 1e9
    
    # FPS测试
    torch_model.eval()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = torch_model(input_tensor)
    
    # 正式测量
    torch.cuda.synchronize()
    start.record()
    with torch.no_grad():
        for _ in range(test_round):
            _ = torch_model(input_tensor)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_time = start.elapsed_time(end) / 1000
    fps = test_round / elapsed_time
    
    return None, None, fps

# 使用示例 --------------------------------------------------
if __name__ == "__main__":
    from ultralytics import YOLO
    
    # 初始化模型（关键修改：使用task参数）
    cfg = "/home/cigit/exp/surface_detect/yolo11l.yaml"
    model = YOLO(cfg, task='detect')  # 显式指定任务类型
    
    # 获取指标
    params, gflops, fps = get_model_metrics(model)
    
    # print(f"Params: {params:.2f}M")
    # print(f"GFLOPs: {gflops:.2f}G")
    print(f"FPS: {fps:.1f}")