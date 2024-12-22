import cv2
import numpy as np
import os
from pix2point.feature_matcher import extract_and_match_features_multiple
from pix2point.reconstruction import triangulate_points_multiple
from pix2point.visualization import visualize_point_cloud_open3d,visualize_point_cloud_pyvista,visualize_point_cloud_matplotlib

# Directory containing the images
image_dir = 'data'

# Automatically find all image files matching the pattern "imageX.png"
image_files = sorted(
    [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.startswith('image') and f.endswith('.png')],
    key=lambda x: int(x.split('image')[-1].split('.')[0])  # Sort by the number in the filename
)

# Load all images
images = [cv2.imread(img) for img in image_files]

# Check if any images were loaded
if len(images) < 2:
    raise ValueError("At least two images are required for 3D reconstruction.")

# Intrinsic camera matrix (example, replace with actual calibration results)
mtx = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]])

# Extract and match features across all images
keypoints, matches = extract_and_match_features_multiple(images)

# Reconstruct the 3D point cloud
points_3d = triangulate_points_multiple(images, keypoints, matches, mtx)

# Visualize the 3D point cloud
visualize_point_cloud_matplotlib(points_3d)
