import torch
from ultralytics import YOLO
import os
import re
from collections import defaultdict

def load_pretrained_weights_to_custom_model(custom_yaml_path, pretrained_weights_path, verbose=True):
    """
    Load pretrained weights into a custom YOLO model, but only for layers that exist in both models.
    Custom modules will remain randomly initialized.
    
    Args:
        custom_yaml_path (str): Path to the custom model YAML file (e.g., 'DASC-YOLO-modified.yaml')
        pretrained_weights_path (str): Path to the pretrained model weights (e.g., 'yolov5n6u.pt')
        verbose (bool): Whether to print detailed information about weight loading
    
    Returns:
        model (YOLO): Custom model with pretrained weights loaded for matching layers
    """
    print(f"Creating custom model from {custom_yaml_path}...")
    custom_model = YOLO(custom_yaml_path)
    
    print(f"Loading pretrained weights from {pretrained_weights_path}...")
    pretrained_model = YOLO(pretrained_weights_path)
    
    # Get state dictionaries for both models
    custom_state_dict = custom_model.model.state_dict()
    pretrained_state_dict = pretrained_model.model.state_dict()
    
    # Create a new state dictionary that combines pretrained weights for existing layers
    # and keeps random initialization for new layers
    new_state_dict = {}
    
    # Track which layers were loaded from pretrained model and which were left random
    loaded_layers = []
    random_init_layers = []
    
    # Statistics for different types of layers
    stats = defaultdict(lambda: {"loaded": 0, "random": 0})
    
    # Identify custom modules from the YAML file to exclude them from weight loading
    custom_modules = ["CBAM", "ProtoNet", "CrossScaleAttention", "DefectDetect"]
    
    def safe_tensor_copy(tensor):
        """Helper function to safely copy tensors and set requires_grad only for float tensors"""
        copied_tensor = tensor.clone().detach()
        if copied_tensor.dtype in [torch.float16, torch.float32, torch.float64]:
            copied_tensor.requires_grad_(True)
        return copied_tensor
    
    # Iterate through the state dictionaries and copy weights where possible
    for key in custom_state_dict:
        # Skip layers from custom modules (they should remain randomly initialized)
        layer_is_custom = any(module in key for module in custom_modules)
        
        # Determine layer type for statistics
        layer_type = "unknown"
        if "conv" in key.lower():
            layer_type = "conv"
        elif "bn" in key.lower():
            layer_type = "batchnorm"
        elif "bias" in key.lower():
            layer_type = "bias"
        elif "weight" in key.lower():
            layer_type = "weight"
        
        # If the layer is from a custom module, keep random initialization
        if layer_is_custom:
            new_state_dict[key] = safe_tensor_copy(custom_state_dict[key])
            random_init_layers.append(key)
            stats[layer_type]["random"] += 1
            continue
            
        # Check if the key exists in the pretrained model and has the same shape
        if key in pretrained_state_dict and custom_state_dict[key].shape == pretrained_state_dict[key].shape:
            new_state_dict[key] = safe_tensor_copy(pretrained_state_dict[key])
            loaded_layers.append(key)
            stats[layer_type]["loaded"] += 1
        else:
            # Try to find a matching layer with a different name but same shape
            matching_key = None
            for pretrained_key in pretrained_state_dict:
                if (custom_state_dict[key].shape == pretrained_state_dict[pretrained_key].shape and
                    key.split('.')[-1] == pretrained_key.split('.')[-1]):  # Match parameter name (weight/bias)
                    matching_key = pretrained_key
                    break
            
            if matching_key:
                new_state_dict[key] = safe_tensor_copy(pretrained_state_dict[matching_key])
                loaded_layers.append(key)
                stats[layer_type]["loaded"] += 1
                if verbose:
                    print(f"Matched {key} to {matching_key} based on shape")
            else:
                new_state_dict[key] = safe_tensor_copy(custom_state_dict[key])
                random_init_layers.append(key)
                stats[layer_type]["random"] += 1
    
    # Load the combined weights into the custom model
    custom_model.model.load_state_dict(new_state_dict)
    
    # Make sure all parameters are leaf tensors
    for param in custom_model.model.parameters():
        if not param.is_leaf and param.dtype in [torch.float16, torch.float32, torch.float64]:
            param.data = param.data.clone().detach().requires_grad_(True)
    
    # Print statistics
    print(f"\nLoaded weights for {len(loaded_layers)} layers from pretrained model")
    print(f"Kept random initialization for {len(random_init_layers)} layers")
    
    # Print detailed statistics by layer type
    print("\nStatistics by layer type:")
    for layer_type, counts in stats.items():
        if counts["loaded"] + counts["random"] > 0:
            print(f"  {layer_type}: {counts['loaded']} loaded, {counts['random']} random")
    
    if verbose:
        print("\nRandomly initialized layers:")
        for layer in sorted(random_init_layers):
            print(f"  {layer}")
    
    return custom_model

def save_model_with_loaded_weights(model, output_path):
    """
    Save the model with loaded weights to a file in a format that can be loaded directly by YOLO()
    
    Args:
        model (YOLO): The model with loaded weights
        output_path (str): Path to save the model to
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    try:
        # 创建与YOLO类兼容的保存格式
        yaml_path = model.yaml if hasattr(model, 'yaml') else 'default.yaml'
        ckpt = {
            'epoch': -1,
            'model': model.model,  # 保存整个模型，而不仅仅是state_dict
            'names': model.names,
            'yaml': yaml_path,
            'ema': None,
            'updates': 0
        }
        
        torch.save(ckpt, output_path)
        print(f"Model saved in YOLO compatible format to {output_path}")
        print(f"You can now load it with YOLO('{output_path}')")
    except Exception as e2:
        print(f"Error saving in compatible format: {e2}")
        
        # 3. 退回到保存state_dict格式，这不能直接用YOLO()加载但可以用于后续处理
        try:
            torch.save({
                'state_dict': model.model.state_dict(),
                'yaml_file': model.yaml if hasattr(model, 'yaml') else None,
                'names': model.names
            }, output_path)
            print(f"Model state_dict saved to {output_path}")
            print("WARNING: This format requires additional processing before it can be loaded with YOLO()")
        except Exception as e3:
            print(f"All saving methods failed. Last error: {e3}")
            raise

if __name__ == "__main__":
    # Example usage
    custom_yaml_path = "DASC-YOLO-modified.yaml"
    pretrained_weights_path = "E:\\Thesis_Master\\yolov5++\\yolov5n6u.pt"  # Path to official weights
    output_path = "custom_weights.pt"
    
    # Check if files exist
    if not os.path.exists(custom_yaml_path):
        raise FileNotFoundError(f"Custom YAML file not found: {custom_yaml_path}")
    
    if not os.path.exists(pretrained_weights_path):
        raise FileNotFoundError(f"Pretrained weights file not found: {pretrained_weights_path}")
    
    # Load model with pretrained weights
    model = load_pretrained_weights_to_custom_model(custom_yaml_path, pretrained_weights_path, verbose=True)
    
    # Save the model
    save_model_with_loaded_weights(model, output_path)
    
    print("\nModel loaded and saved successfully!")
    
    # Example inference (uncomment to use)
    # results = model("path/to/image.jpg") 