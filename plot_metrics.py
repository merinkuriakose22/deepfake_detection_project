import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from image_comparator import compare_images

# Paths to extracted frames
real_frames_dir = 'frames/real_frames'
fake_frames_dir = 'frames/fake_frames'

real_frames = sorted(os.listdir(real_frames_dir))
fake_frames = sorted(os.listdir(fake_frames_dir))

# Ensure same number of frames
num_frames = min(len(real_frames), len(fake_frames))

if num_frames == 0:
    print("No frames found in one or both folders.")
    exit()

mse_values = []
ssim_values = []

for i in range(num_frames):
    real_frame_path = os.path.join(real_frames_dir, real_frames[i])
    fake_frame_path = os.path.join(fake_frames_dir, fake_frames[i])

    imageA = cv2.imread(real_frame_path)
    imageB = cv2.imread(fake_frame_path)

    if imageA is None or imageB is None:
        print(f"Error loading frames: {real_frame_path} or {fake_frame_path}")
        continue

    if imageA.shape != imageB.shape:
        imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]))

    mse, ssim = compare_images(imageA, imageB)

    mse_values.append(mse)
    ssim_values.append(ssim)

# Plot MSE and SSIM trends
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(mse_values, marker='o')
plt.title('MSE across frames')
plt.xlabel('Frame Index')
plt.ylabel('MSE')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(ssim_values, marker='o', color='green')
plt.title('SSIM across frames')
plt.xlabel('Frame Index')
plt.ylabel('SSIM')
plt.grid()

plt.tight_layout()
plt.show()
