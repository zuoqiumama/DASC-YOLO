import os
from PIL import Image

def split_images(input_dir, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        # 检查常见图片格式
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            input_path = os.path.join(input_dir, filename)
            
            try:
                with Image.open(input_path) as img:
                    # 验证图片尺寸
                    if img.size != (8192, 2048):
                        print(f"跳过 {filename}: 尺寸不符合要求（应为8192x2048）")
                        continue

                    # 准备文件名组件
                    base_name = os.path.splitext(filename)[0]
                    extension = os.path.splitext(filename)[1]

                    # 切割并保存四个图块
                    for i in range(4):
                        # 计算切割区域（左，上，右，下）
                        left = i * 2048
                        right = left + 2048
                        box = (left, 0, right, 2048)

                        # 切割并保存
                        crop_img = img.crop(box)
                        output_filename = f"{base_name}_part{i+1}{extension}"
                        output_path = os.path.join(output_dir, output_filename)
                        crop_img.save(output_path)
                        
                    print(f"处理完成：{filename} → 4个图块")

            except Exception as e:
                print(f"处理 {filename} 时出错：{str(e)}")

if __name__ == "__main__":
    # 配置路径（按需修改）
    input_directory = "test"   # 原始图片存放路径
    output_directory = "split"

    split_images(input_directory, output_directory)
    print("所有图片处理完成！")