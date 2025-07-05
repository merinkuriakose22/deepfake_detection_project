import os
from frame_extractor import extract_frames
from image_comparator import compare_images
import cv2

# Base and dataset directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'SDFVD Small-scale Deepfake Forgery Video Dataset')

# Correct paths to videos
REAL_DIR = os.path.join(DATASET_DIR, 'SDFVD', 'videos_real')
FAKE_DIR = os.path.join(DATASET_DIR, 'SDFVD', 'videos_fake')

# Frame storage directories
REAL_FRAMES_DIR = os.path.join(BASE_DIR, 'frames', 'real_frames')
FAKE_FRAMES_DIR = os.path.join(BASE_DIR, 'frames', 'fake_frames')

# Create frame directories if they don't exist
os.makedirs(REAL_FRAMES_DIR, exist_ok=True)
os.makedirs(FAKE_FRAMES_DIR, exist_ok=True)

# Extract frames from real videos
real_videos = sorted(os.listdir(REAL_DIR))
for video in real_videos:
    video_path = os.path.join(REAL_DIR, video)
    extract_frames(video_path, REAL_FRAMES_DIR)

# Extract frames from fake videos
fake_videos = sorted(os.listdir(FAKE_DIR))
for video in fake_videos:
    video_path = os.path.join(FAKE_DIR, video)
    extract_frames(video_path, FAKE_FRAMES_DIR)

print("Frame extraction completed.")

# Compare one frame from each set
real_frame_path = os.path.join(REAL_FRAMES_DIR, '0.jpg')
fake_frame_path = os.path.join(FAKE_FRAMES_DIR, '0.jpg')

imageA = cv2.imread(real_frame_path)
imageB = cv2.imread(fake_frame_path)

if imageA is None or imageB is None:
    print("Error loading images for comparison. Please check frame extraction.")
else:
    mse_value, ssim_value = compare_images(imageA, imageB)
    print(f"MSE: {mse_value:.2f}")
    print(f"SSIM: {ssim_value:.2f}")
