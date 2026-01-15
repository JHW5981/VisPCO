import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import os
from pathlib import Path

# 目标面积范围
TARGET_MIN_AREA = 400
TARGET_MAX_AREA = 1024*1024

def calculate_target_size(original_w, original_h, target_area):
    """
    根据目标面积和原始长宽比计算新尺寸
    保留原始长宽比
    """
    aspect_ratio = original_w / original_h
    
    if aspect_ratio >= 1:  # 宽 >= 高
        new_h = int(np.sqrt(target_area / aspect_ratio))
        new_w = int(new_h * aspect_ratio)
    else:  # 高 > 宽
        new_w = int(np.sqrt(target_area * aspect_ratio))
        new_h = int(new_w / aspect_ratio)
    
    return new_w, new_h

def redistribute_areas_uniformly(area_list, target_min=TARGET_MIN_AREA, target_max=TARGET_MAX_AREA):
    """
    将原始面积列表重新映射到目标范围内均匀分布
    使用分位数映射方法
    """
    area_array = np.array(area_list)
    sorted_indices = np.argsort(area_array)
    n = len(area_array)
    target_areas = np.linspace(target_min, target_max, n)
    
    target_area_map = {}
    for idx, original_idx in enumerate(sorted_indices):
        target_area_map[original_idx] = target_areas[idx]
    
    return target_area_map

# 加载数据
print("正在加载数据...")
data = []
with open("/mnt/inaisfs/home/test3/jihuawei/workspace/Dynamic-VLM-Single-Node-Qwen/created_datasets/eval_downsample_ratio_1_dataset.json", "r") as f:
    data1 = json.load(f)
    for d in data1.keys():
        data.extend(data1[d])

print(f"数据集大小: {len(data)}")

# 计算原始面积
print("正在计算原始面积...")
original_areas = []
original_sizes = []
for d in tqdm(data, desc="计算原始面积"):
    img = Image.open(d["images"][0])
    w, h = img.size
    area = w * h
    original_areas.append(area)
    original_sizes.append((w, h))

original_areas = np.array(original_areas)
print(f"原始面积: 最小={original_areas.min()}, 最大={original_areas.max()}, 中位数={np.median(original_areas):.0f}")

# 计算目标面积映射
print("正在计算目标面积映射...")
target_area_map = redistribute_areas_uniformly(original_areas, TARGET_MIN_AREA, TARGET_MAX_AREA)

# 设置输出目录
output_base_dir = "/mnt/inaisfs/home/test3/jihuawei/pretraining_data/resize_dataset"
os.makedirs(output_base_dir, exist_ok=True)

# 调整图像尺寸并保存
print("正在调整图像尺寸并保存...")
new_areas = []
for idx, d in enumerate(tqdm(data, desc="处理图像")):
    original_img_path = d["images"][0]
    w, h = original_sizes[idx]
    target_area = target_area_map[idx]
    new_w, new_h = calculate_target_size(w, h, target_area)
    new_area = new_w * new_h
    new_areas.append(new_area)
    
    # 打开原始图像
    img = Image.open(original_img_path)
    
    # 调整尺寸
    resized_img = img.resize((new_w, new_h), Image.LANCZOS)
    
    # 构建输出路径，保留一级目录结构
    original_path = Path(original_img_path)
    
    # 从原始路径中提取一级目录名（images/后面的第一个目录）
    # 例如：/mnt/.../LMUData/images/A-OKVQA/219.jpg -> A-OKVQA
    path_parts = original_path.parts
    try:
        # 找到 'images' 在路径中的位置
        images_idx = path_parts.index('images')
        if images_idx + 1 < len(path_parts):
            first_level_dir = path_parts[images_idx + 1]
        else:
            # 如果没有找到images目录，使用父目录名
            first_level_dir = original_path.parent.name
    except ValueError:
        # 如果路径中没有'images'，使用父目录名
        first_level_dir = original_path.parent.name
    
    # 创建输出目录（包含一级目录）
    output_dir = Path(output_base_dir) / first_level_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建完整输出路径
    output_path = output_dir / original_path.name
    
    # 保存调整后的图像
    resized_img.save(output_path, quality=95)

new_areas = np.array(new_areas)
print(f"\n调整后面积统计:")
print(f"  最小值: {new_areas.min()}")
print(f"  最大值: {new_areas.max()}")
print(f"  中位数: {np.median(new_areas):.0f}")
print(f"  平均值: {new_areas.mean():.0f}")
print(f"\n所有图像已保存到: {output_base_dir}")
