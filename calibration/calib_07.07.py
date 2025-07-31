import cv2
import numpy as np
import os
import glob

# Paths
input_dir = '/home/wego/Pictures/captures/new_sinkhole/'
output_dir = '/home/wego/Pictures/captures/new_sinkhole_calib/'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load calibration parameters
with np.load('0701.npz') as data:
    K = data['K']
    D = data['D']

# Get all PNG files
image_paths = glob.glob(os.path.join(input_dir, '*.png'))

for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read {img_path}")
        continue

    h, w = img.shape[:2]

    # Undistort map
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
    )
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

    # Save the undistorted image
    filename = os.path.basename(img_path)
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, undistorted_img)

    print(f"Saved undistorted image: {save_path}")
