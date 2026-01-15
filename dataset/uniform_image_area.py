import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import os
from pathlib import Path
import argparse

# Target area range
TARGET_MIN_AREA = 400
TARGET_MAX_AREA = 1024 * 1024

def calculate_target_size(original_w, original_h, target_area):
    """
    Calculate new dimensions based on the target area and original aspect ratio.
    Preserve the original aspect ratio.
    """
    aspect_ratio = original_w / original_h

    if aspect_ratio >= 1:  # width >= height
        new_h = int(np.sqrt(target_area / aspect_ratio))
        new_w = int(new_h * aspect_ratio)
    else:  # height > width
        new_w = int(np.sqrt(target_area * aspect_ratio))
        new_h = int(new_w / aspect_ratio)

    return new_w, new_h

def redistribute_areas_uniformly(area_list, target_min=TARGET_MIN_AREA, target_max=TARGET_MAX_AREA):
    """
    Map the original area list uniformly into the target range
    using quantile mapping.
    """
    area_array = np.array(area_list)
    sorted_indices = np.argsort(area_array)
    n = len(area_array)
    target_areas = np.linspace(target_min, target_max, n)

    target_area_map = {}
    for idx, original_idx in enumerate(sorted_indices):
        target_area_map[original_idx] = target_areas[idx]

    return target_area_map

def resize_and_update_json(
    json_input_path,
    output_base_dir,
    json_output_path=None,
    target_min_area=TARGET_MIN_AREA,
    target_max_area=TARGET_MAX_AREA
):
    """
    Resize all images in the dataset described by the input JSON,
    save them to the output directory, and update the image paths in the JSON.

    Args:
        json_input_path (str): Path to the input JSON file.
        output_base_dir (str): Directory to save resized images.
        json_output_path (str): Path to save the updated JSON file. If None, appends '_resized.json'.
        target_min_area (int): Minimum area for target resizing.
        target_max_area (int): Maximum area for target resizing.

    Returns:
        data (list): The updated data list after resizing images and updating paths.
    """
    print("Loading data...")
    with open(json_input_path, "r") as f:
        data1 = json.load(f)
        # data1 is assumed to be a dict of lists
        data = []
        for d in data1.keys():
            data.extend(data1[d])

    print(f"Dataset size: {len(data)}")

    # Calculate original areas and sizes
    print("Calculating original areas...")
    original_areas = []
    original_sizes = []
    for d in tqdm(data, desc="Calculating original areas"):
        img = Image.open(d["images"][0])
        w, h = img.size
        area = w * h
        original_areas.append(area)
        original_sizes.append((w, h))

    original_areas = np.array(original_areas)
    print(f"Original area: min={original_areas.min()}, max={original_areas.max()}, median={np.median(original_areas):.0f}")

    # Calculate target area mapping
    print("Calculating target area mapping...")
    target_area_map = redistribute_areas_uniformly(original_areas, target_min_area, target_max_area)

    # Set output directory
    os.makedirs(output_base_dir, exist_ok=True)

    # Resize images, save, and update JSON image path
    print("Resizing images and saving...")
    new_areas = []
    for idx, d in enumerate(tqdm(data, desc="Processing images")):
        original_img_path = d["images"][0]
        w, h = original_sizes[idx]
        target_area = target_area_map[idx]
        new_w, new_h = calculate_target_size(w, h, target_area)
        new_area = new_w * new_h
        new_areas.append(new_area)

        # Open the original image
        img = Image.open(original_img_path)

        # Resize the image
        resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Build output path, preserving the first-level directory structure
        original_path = Path(original_img_path)
        path_parts = original_path.parts
        try:
            # Find 'images' in the path
            images_idx = path_parts.index('images')
            if images_idx + 1 < len(path_parts):
                first_level_dir = path_parts[images_idx + 1]
            else:
                first_level_dir = original_path.parent.name
        except ValueError:
            first_level_dir = original_path.parent.name

        # Create output directory (including first-level directory)
        output_dir = Path(output_base_dir) / first_level_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Construct the full output path
        output_path = output_dir / original_path.name

        # Save the resized image
        resized_img.save(output_path, quality=95)

        # Update the image path in the JSON to the new one (as string)
        d["images"][0] = str(output_path)

    new_areas = np.array(new_areas)
    print(f"\nStatistics after resizing:")
    print(f"  min: {new_areas.min()}")
    print(f"  max: {new_areas.max()}")
    print(f"  median: {np.median(new_areas):.0f}")
    print(f"  mean: {new_areas.mean():.0f}")
    print(f"\nAll images have been saved to: {output_base_dir}")

    # Save updated JSON if requested
    if json_output_path is not None:
        # Try to restore the original grouping if possible
        output_dict = {}
        with open(json_input_path, "r") as f:
            orig_data1 = json.load(f)
            group_keys = list(orig_data1.keys())
            current_idx = 0
            for k in group_keys:
                group_len = len(orig_data1[k])
                output_dict[k] = data[current_idx:current_idx + group_len]
                current_idx += group_len
        with open(json_output_path, "w") as outf:
            json.dump(output_dict, outf, ensure_ascii=False, indent=2)
        print(f"\nUpdated JSON has been saved to: {json_output_path}")

    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uniformly resize images in a dataset JSON with controlled area range.")
    parser.add_argument("--json_input_path", type=str, required=True,
                        help="Path to the input JSON file.")
    parser.add_argument("--output_base_dir", type=str, required=True,
                        help="Directory to save resized images.")
    parser.add_argument("--json_output_path", type=str, default=None,
                        help="Path to save the updated JSON file (optional).")
    parser.add_argument("--target_min_area", type=int, default=TARGET_MIN_AREA,
                        help=f"Minimum area for resizing (default {TARGET_MIN_AREA}).")
    parser.add_argument("--target_max_area", type=int, default=TARGET_MAX_AREA,
                        help=f"Maximum area for resizing (default {TARGET_MAX_AREA}).")

    args = parser.parse_args()

    resize_and_update_json(
        json_input_path=args.json_input_path,
        output_base_dir=args.output_base_dir,
        json_output_path=args.json_output_path,
        target_min_area=args.target_min_area,
        target_max_area=args.target_max_area
    )
