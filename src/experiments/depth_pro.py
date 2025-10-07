import os
import glob
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
from PIL import Image
import torch
import numpy as np

# Set up paths
project_root = "."  # Adjust this to your project root
images_dir = os.path.join(project_root, "assets", "colmap", "images")
depth_maps_dir = os.path.join(project_root, "assets", "colmap", "depth_maps")
focal_lengths_path = os.path.join(depth_maps_dir, "focal_lengths.npy")

# Create focal lengths dictionary
focal_lengths = {}

# Create depth maps directory if it doesn't exist
os.makedirs(depth_maps_dir, exist_ok=True)

# Load processor and model
print("Loading model...")
processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Get all image files (common formats)
image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    image_files.extend(glob.glob(os.path.join(images_dir, ext.upper())))

print(f"Found {len(image_files)} images to process")

# Process each image
for i, image_path in enumerate(image_files, 1):
    try:
        # Load image
        image = Image.open(image_path)

        # Get filename without extension for output naming
        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]
        output_path = os.path.join(depth_maps_dir, f"{name_without_ext}_map.jpg")

        # Skip if depth map already exists (check for .npy file)
        numpy_output_path = os.path.join(depth_maps_dir, f"{name_without_ext}_map.npy")
        if os.path.exists(numpy_output_path):
            print(
                f"[{i}/{len(image_files)}] Skipping {filename} (depth map already exists)"
            )
            continue

        # Prepare input
        inputs = processor(images=image, return_tensors="pt").to(model.device)

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process to get depth map
        post = processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.height, image.width)]
        )
        depth_map = post[0]["predicted_depth"]
        focal_length = post[0]["focal_length"]
        focal_lengths[filename] = focal_length.item()

        # Convert depth map to numpy array (preserve exact metric values)
        depth_array = depth_map.squeeze().cpu().detach().numpy()

        # Save raw depth values as numpy array
        numpy_output_path = os.path.join(depth_maps_dir, f"{name_without_ext}_map.npy")
        np.save(numpy_output_path, depth_array)

        # Also save as 16-bit TIFF to preserve precision while being image-readable
        # Scale to 16-bit range while preserving relative metric values
        depth_scaled = (depth_array * 1000).astype(
            np.uint16
        )  # Convert meters to millimeters
        depth_image = Image.fromarray(depth_scaled, mode="I;16")
        tiff_output_path = os.path.join(depth_maps_dir, f"{name_without_ext}_map.tiff")
        depth_image.save(tiff_output_path)

        print(
            f"[{i}/{len(image_files)}] Processed {filename} -> {name_without_ext}_map.npy/.tiff"
        )

    except Exception as e:
        print(
            f"[{i}/{len(image_files)}] Error processing {os.path.basename(image_path)}: {str(e)}"
        )
        continue

# Save focal lengths to a numpy file
np.save(focal_lengths_path, focal_lengths)

print(f"\nCompleted! Processed {len(image_files)} images.")
print(f"Depth maps saved to: {depth_maps_dir}")
print(f"Saved focal lengths for {len(focal_lengths)} images to focal_lengths.npy")
