import pickle
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# 读取 .pk 文件
split = "Mix-train"
pk_file = f"{split}.pk"  # 请替换成你的文件名
output_dir = f"{split}"  # 保存图片的文件夹
os.makedirs(output_dir, exist_ok=True)

# 读取数据
with open(pk_file, "rb") as f:
    data = pickle.load(f)

# 遍历所有数据并保存图像
for idx, item in enumerate(data):
    if "image" in item:
        image_data = item["image"]

        # 处理 NumPy 数组格式的图像数据
        if isinstance(image_data, np.ndarray):
            img = Image.fromarray(image_data)
        elif isinstance(image_data, list):
            # 如果是列表，转换为 NumPy 数组
            img = Image.fromarray(np.array(image_data, dtype=np.uint8))
        else:
            print(f"Skipping index {idx}: Unsupported image format")
            continue  # 跳过无法处理的数据

        # 生成文件名（使用 ID 作为文件名，如果有 "id" 字段）
        file_name = f"{idx}.png"
        if "id" in item:
            file_name = f"{item['id']}.png"

        # 保存图像
        img_path = os.path.join(output_dir, file_name)
        img.save(img_path)
        print(f"Saved {img_path}")

print("All images have been extracted and saved!")
