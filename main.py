import os
import cv2
import csv
import matplotlib.pyplot as plt
from frame_extractor import extract_frames
from image_comparator import compare_images

DATA_DIR = r'C:\Users\TEMP.MERIN\OneDrive\Desktop\deepfake_detection_project\SDFVD Small-scale Deepfake Forgery Video Dataset\SDFVD'
REAL_DIR = os.path.join(DATA_DIR, 'videos_real')
FAKE_DIR = os.path.join(DATA_DIR, 'videos_fake')

OUTPUT_BASE = 'output'
REAL_FRAMES = os.path.join(OUTPUT_BASE, 'real_frames')
FAKE_FRAMES = os.path.join(OUTPUT_BASE, 'fake_frames')
CSV_LOG = os.path.join(OUTPUT_BASE, 'frame_comparison_log.csv')

os.makedirs(REAL_FRAMES, exist_ok=True)
os.makedirs(FAKE_FRAMES, exist_ok=True)

real_videos = sorted(os.listdir(REAL_DIR))
fake_videos = sorted(os.listdir(FAKE_DIR))

csv_headers = ['Video_Name', 'Frame_Number', 'MSE', 'SSIM', 'Classification']

with open(CSV_LOG, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_headers)

    for rv, fv in zip(real_videos, fake_videos):
        real_path = os.path.join(REAL_DIR, rv)
        fake_path = os.path.join(FAKE_DIR, fv)

        print(f"\nProcessing: {rv} vs {fv}")

        extract_frames(real_path, REAL_FRAMES)
        extract_frames(fake_path, FAKE_FRAMES)

        real_files = sorted(os.listdir(REAL_FRAMES))
        fake_files = sorted(os.listdir(FAKE_FRAMES))
        total_frames = min(len(real_files), len(fake_files))

        mse_values = []
        ssim_values = []
        classifications = []

        for i in range(1, total_frames + 1):
            img_r = cv2.imread(f"{REAL_FRAMES}/{i}.jpg", cv2.IMREAD_GRAYSCALE)
            img_f = cv2.imread(f"{FAKE_FRAMES}/{i}.jpg", cv2.IMREAD_GRAYSCALE)

            if img_r.shape != img_f.shape:
                img_f = cv2.resize(img_f, (img_r.shape[1], img_r.shape[0]))

            mse_val, ssim_val = compare_images(img_f, img_r, f"{rv} vs {fv} - Frame {i}")
            mse_values.append(mse_val)
            ssim_values.append(ssim_val)

            if ssim_val < 0.85:
                classification = "Fake"
            else:
                classification = "Real"

            classifications.append(classification)
            print(f"Frame {i}: MSE={mse_val:.2f}, SSIM={ssim_val:.2f}, Classified as: {classification}")

            writer.writerow([f"{rv} vs {fv}", i, f"{mse_val:.2f}", f"{ssim_val:.2f}", classification])

        
        frames = list(range(1, total_frames + 1))
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(frames, mse_values, marker='o', color='blue')
        plt.title(f"MSE Trend - {rv} vs {fv}")
        plt.xlabel("Frame Number")
        plt.ylabel("MSE")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(frames, ssim_values, marker='o', color='green')
        plt.axhline(y=0.85, color='red', linestyle='--', label='SSIM Threshold')
        plt.title(f"SSIM Trend - {rv} vs {fv}")
        plt.xlabel("Frame Number")
        plt.ylabel("SSIM")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        for f in os.listdir(REAL_FRAMES):
            os.remove(os.path.join(REAL_FRAMES, f))
        for f in os.listdir(FAKE_FRAMES):
            os.remove(os.path.join(FAKE_FRAMES, f))

print(f"\nCSV log has been saved to {CSV_LOG}")
